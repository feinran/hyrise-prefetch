#pragma once

#include <cstddef>
#include <cstdlib>
#include <memory>
#include <utility>
#include <vector>

#include "storage/pos_lists/abstract_pos_list.hpp"
#include "storage/segment_iterables.hpp"
#include "storage/value_segment.hpp"

namespace hyrise {

namespace WIP {
// Reads PREFETCH_DISTANCE from the environment exactly once per program run.
// Using an inline function ensures a single static instance across all translation units.
inline size_t prefetch_distance_from_env() {
  static const size_t distance = [] {
    if (const char* env_var = std::getenv("PREFETCH_DISTANCE")) {
      auto distance = static_cast<size_t>(std::stoi(env_var));
      std::cerr << "Read distance: " << distance << '\n';
      return distance;
    }
    size_t default_distance = 32;
    std::cerr << "Using default distance: " << default_distance << '\n';
    return default_distance;
  }();
  return distance;
}

template <typename ValueVectorIterator, typename PosListIteratorType>
inline void prefetch_value(ValueVectorIterator values_begin_it, PosListIteratorType position_filter_it,
                           PosListIteratorType position_filter_end) {
  const size_t prefetch_distance = 32; // prefetch_distance_from_env();
  const size_t distance_to_end = position_filter_end - position_filter_it;
  const size_t prefetch_distance_clamped = std::min(distance_to_end - 1, prefetch_distance);
  const auto prefetch_offset = (position_filter_it + prefetch_distance_clamped)->chunk_offset;
  const auto prefetch_address = std::to_address(values_begin_it + prefetch_offset);
  __builtin_prefetch(prefetch_address, 0, 3);
}
}  // namespace WIP

// Our function naming for iterables is not correct. `_on_with` is a public function and should not start with `_`,
// whereas the iterator functions should start with `_` as they are private (but can't, because boost requires them).
// NOLINTBEGIN(readability-identifier-naming)
template <typename T>
class ValueSegmentIterable : public PointAccessibleSegmentIterable<ValueSegmentIterable<T>> {
 public:
  using ValueType = T;

  explicit ValueSegmentIterable(const ValueSegment<T>& segment) : _segment{segment} {}

  template <typename Functor>
  void _on_with_iterators(const Functor& functor) const {
    _segment.access_counter[SegmentAccessCounter::AccessType::Sequential] += _segment.size();
    if (_segment.is_nullable()) {
      auto begin = Iterator{_segment.values().cbegin(), _segment.values().cbegin(), _segment.null_values().cbegin()};
      auto end = Iterator{_segment.values().cbegin(), _segment.values().cend(), _segment.null_values().cend()};
      functor(begin, end);
    } else {
      auto begin = NonNullIterator{_segment.values().cbegin(), _segment.values().cbegin()};
      auto end = NonNullIterator{_segment.values().cbegin(), _segment.values().cend()};
      functor(begin, end);
    }
  }

  template <typename Functor, typename PosListType>
  void _on_with_iterators(const std::shared_ptr<PosListType>& position_filter, const Functor& functor) const {
    _segment.access_counter[SegmentAccessCounter::access_type(*position_filter)] += position_filter->size();

    using PosListIteratorType = std::decay_t<decltype(position_filter->cbegin())>;

    if (_segment.is_nullable()) {
      auto begin = PointAccessIterator<PosListIteratorType>{_segment.values().cbegin(), _segment.null_values().cbegin(),
                                                            position_filter->cbegin(), position_filter->cbegin(),
                                                            position_filter->cend()};
      auto end = PointAccessIterator<PosListIteratorType>{_segment.values().cbegin(), _segment.null_values().cbegin(),
                                                          position_filter->cbegin(), position_filter->cend(),
                                                          position_filter->cend()};
      functor(begin, end);
    } else {
      auto begin = NonNullPointAccessIterator<PosListIteratorType>{
          _segment.values().cbegin(), position_filter->cbegin(), position_filter->cbegin(), position_filter->cend()};
      auto end = NonNullPointAccessIterator<PosListIteratorType>{_segment.values().cbegin(), position_filter->cbegin(),
                                                                 position_filter->cend(), position_filter->cend()};
      functor(begin, end);
    }
  }

  size_t _on_size() const {
    return _segment.size();
  }

 private:
  const ValueSegment<T>& _segment;

  class NonNullIterator : public AbstractSegmentIterator<NonNullIterator, NonNullSegmentPosition<T>> {
   public:
    using ValueType = T;
    using IterableType = ValueSegmentIterable<T>;
    using ValueIterator = typename pmr_vector<T>::const_iterator;

    explicit NonNullIterator(ValueIterator begin_value_it, ValueIterator value_it)
        : _value_it{std::move(value_it)},
          _chunk_offset{static_cast<ChunkOffset>(std::distance(begin_value_it, _value_it))} {}

   private:
    friend class boost::iterator_core_access;  // grants the boost::iterator_facade access to the private interface

    void increment() {
      ++_value_it;
      ++_chunk_offset;
    }

    void decrement() {
      --_value_it;
      --_chunk_offset;
    }

    void advance(std::ptrdiff_t distance) {
      _value_it += distance;
      _chunk_offset += distance;
    }

    bool equal(const NonNullIterator& other) const {
      return _value_it == other._value_it;
    }

    std::ptrdiff_t distance_to(const NonNullIterator& other) const {
      return other._value_it - _value_it;
    }

    NonNullSegmentPosition<T> dereference() const {
      return NonNullSegmentPosition<T>{*_value_it, _chunk_offset};
    }

    ValueIterator _value_it;
    ChunkOffset _chunk_offset;
  };

  class Iterator : public AbstractSegmentIterator<Iterator, SegmentPosition<T>> {
   public:
    using ValueType = T;
    using IterableType = ValueSegmentIterable<T>;
    using ValueIterator = typename pmr_vector<T>::const_iterator;
    using NullValueIterator = pmr_vector<bool>::const_iterator;

    explicit Iterator(ValueIterator begin_value_it, ValueIterator value_it, NullValueIterator null_value_it)
        : _value_it(std::move(value_it)),
          _null_value_it{null_value_it},
          _chunk_offset{static_cast<ChunkOffset>(std::distance(begin_value_it, _value_it))} {}

   private:
    friend class boost::iterator_core_access;  // grants the boost::iterator_facade access to the private interface

    void increment() {
      ++_value_it;
      ++_null_value_it;
      ++_chunk_offset;
    }

    void decrement() {
      --_value_it;
      --_null_value_it;
      --_chunk_offset;
    }

    void advance(std::ptrdiff_t distance) {
      _value_it += distance;
      _null_value_it += distance;
      _chunk_offset += distance;
    }

    bool equal(const Iterator& other) const {
      return _value_it == other._value_it;
    }

    std::ptrdiff_t distance_to(const Iterator& other) const {
      return other._value_it - _value_it;
    }

    SegmentPosition<T> dereference() const {
      return SegmentPosition<T>{*_value_it, *_null_value_it, _chunk_offset};
    }

    ValueIterator _value_it;
    NullValueIterator _null_value_it;
    ChunkOffset _chunk_offset;
  };

  template <typename PosListIteratorType>
  class NonNullPointAccessIterator
      : public AbstractPointAccessSegmentIterator<NonNullPointAccessIterator<PosListIteratorType>, SegmentPosition<T>,
                                                  PosListIteratorType> {
   public:
    using ValueType = T;
    using IterableType = ValueSegmentIterable<T>;
    using ValueVectorIterator = typename pmr_vector<T>::const_iterator;

    explicit NonNullPointAccessIterator(ValueVectorIterator values_begin_it, PosListIteratorType position_filter_begin,
                                        PosListIteratorType position_filter_it, PosListIteratorType position_filter_end)
        : AbstractPointAccessSegmentIterator<NonNullPointAccessIterator, SegmentPosition<T>,
                                             PosListIteratorType>{std::move(position_filter_begin),
                                                                  std::move(position_filter_it)},
          _values_begin_it{std::move(values_begin_it)},
          _position_filter_end{std::move(position_filter_end)} {}

   private:
    friend class boost::iterator_core_access;  // grants the boost::iterator_facade access to the private interface

    SegmentPosition<T> dereference() const {
      const auto& chunk_offsets = this->_chunk_offsets();
      WIP::prefetch_value(_values_begin_it, this->_position_filter_it, _position_filter_end);
      return SegmentPosition<T>{*(_values_begin_it + chunk_offsets.offset_in_referenced_chunk), false,
                                chunk_offsets.offset_in_poslist};
    }

    ValueVectorIterator _values_begin_it;
    PosListIteratorType _position_filter_end;
  };

  template <typename PosListIteratorType>
  class PointAccessIterator : public AbstractPointAccessSegmentIterator<PointAccessIterator<PosListIteratorType>,
                                                                        SegmentPosition<T>, PosListIteratorType> {
   public:
    using ValueType = T;
    using IterableType = ValueSegmentIterable<T>;
    using ValueVectorIterator = typename pmr_vector<T>::const_iterator;
    using NullValueVectorIterator = typename pmr_vector<bool>::const_iterator;

    explicit PointAccessIterator(ValueVectorIterator values_begin_it, NullValueVectorIterator null_values_begin_it,
                                 PosListIteratorType position_filter_begin, PosListIteratorType position_filter_it,
                                 PosListIteratorType position_filter_end)
        : AbstractPointAccessSegmentIterator<PointAccessIterator, SegmentPosition<T>, PosListIteratorType>{
              std::move(position_filter_begin), std::move(position_filter_it)},
          _values_begin_it{std::move(values_begin_it)},
          _null_values_begin_it{null_values_begin_it},
          _position_filter_end{std::move(position_filter_end)} {}

   private:
    friend class boost::iterator_core_access;  // grants the boost::iterator_facade access to the private interface

    SegmentPosition<T> dereference() const {
      const auto& chunk_offsets = this->_chunk_offsets();
      WIP::prefetch_value(_values_begin_it, this->_position_filter_it, _position_filter_end);
      return SegmentPosition<T>{*(_values_begin_it + chunk_offsets.offset_in_referenced_chunk),
                                *(_null_values_begin_it + chunk_offsets.offset_in_referenced_chunk),
                                chunk_offsets.offset_in_poslist};
    }

    ValueVectorIterator _values_begin_it;
    NullValueVectorIterator _null_values_begin_it;
    PosListIteratorType _position_filter_end;
  };
};

// NOLINTEND(readability-identifier-naming)
}  // namespace hyrise
