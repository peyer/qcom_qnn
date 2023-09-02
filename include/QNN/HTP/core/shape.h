//==============================================================================
//
// Copyright (c) 2020 Qualcomm Technologies, Inc.
// All Rights Reserved.
// Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#ifndef HEXNN_SHAPE_H
#define HEXNN_SHAPE_H 1

#include <cstdint>
#include <set>
#include <array>
#include <cstring>
#include <map>
#include "interface_defs.h"
#include "template_help.h"
#include "serialize_defs.h"

class Graph;

// a bit of weirdness here, to avoid the need to use a different std::map or std::set
// for each rank of shape.
// Existing shapes are registered in a multimap<unsigned, void const*>, one per Rank;
// the key is a hash, and the value is a pointer to a Shape<N>.
//
namespace hnnx {

using shape_reduce_map = std::multimap<unsigned, void const *>;
//
// shape_hash is used to hash a Shape<Rank> object; the 'len' value is
// supplied by Shape::shplen(), and depends on Rank.
unsigned shape_hash(void const *, unsigned shplen);
//
// this compares two shapes for equality.
inline bool shape_compare_eq(void const *a, void const *b, unsigned shplen)
{
    return std::memcmp(a, b, shplen) == 0;
}

// This looks up a shape in a shape_reduce_map, independently of Rank.
// If a matching value is found, it returns an iterator to it, and the caller
// can look at iter->second. If no value is found, it inserts an entry
//  { hash, nullptr}, and returns the iterator pointing to that; caller
//  will see iter->second is null, and must replace it with a pointer
//  to a persistent value equal to '*shp'.
shape_reduce_map::iterator shape_find_in_map(shape_reduce_map &map, void const *shp, unsigned hash, unsigned shplen);

void shape_serialize(Serializer &sctx, unsigned rank, size_t const *dims, size_t const *max_dims, uint8_t const *pad);

// values for the ShapeFlags.flags
// 'constant' is used for constant tensors
// 'uncached' is used for 'uncached' (dma spill/fill)
// Note: other flags may be or'd' in later in upper bits,
// so test 'constant' and 'uncached' as bit tests.
enum class ShapeFlag {
    none = 0,
    constant = 1,
    uncached = 2, // mutually exclusive with 'constant'
};

struct ShapeFlags {
    uint16_t flags;
    // must not have any undefined padding between ShapeFlags and Shape<Rank>dims,
    // so we have this explicit paddings
  private:
    uint16_t padding[sizeof(size_t) / sizeof(uint16_t) - 1] = {
            0,
    };

  public:
    ShapeFlags() : flags(0) {}
    explicit ShapeFlags(ShapeFlag flags_in) : flags(unsigned(flags_in)) {}
    ShapeFlags(ShapeFlags const &) = default;

    inline bool is_const_memory() const { return (flags & unsigned(ShapeFlag::constant)) != 0; }
    inline bool is_uncached_memory() const { return (flags & unsigned(ShapeFlag::uncached)) != 0; }
    inline bool ok_src_bypass() const
    {
        return (flags & (unsigned(ShapeFlag::uncached) | unsigned(ShapeFlag::uncached))) != 0;
    }
    inline bool ok_dst_bypass() const { return (flags & unsigned(ShapeFlag::uncached)) != 0; }

    // avoid warning about unused private member:
    unsigned avoid_warning() const { return padding[0]; }
};
// This is used by 'persistent_clone' to duplicate a shape object, but with a new flags value.
// The 'ref_shape' is really a pointer to a Shape<rank>, with 'rank' in supported range.
// This will just do Shape<rank>::canonical_shape( graph, *ref_shape, new_flags), and return the result
// cast back to ShapeFlags const *.
ShapeFlags const *copy_shape_with_flags(Graph &gr, ShapeFlags const *ref_shape, unsigned rank, ShapeFlag newflags);

} // namespace hnnx

#pragma GCC visibility push(default)

template <size_t Rank> struct Shape : public hnnx::ShapeFlags {
    Shape() : dims(), max_dims(), pad(){};
    explicit Shape(const size_t *dims_in)
        : dims(hnnx::ptr_to_stdarray<Rank, size_t>(dims_in)), max_dims(hnnx::ptr_to_stdarray<Rank, size_t>(dims_in)),
          pad(){};
    Shape(std::array<size_t, Rank> dims_in, std::array<size_t, Rank> max_dims_in)
        : dims(dims_in), max_dims(max_dims_in), pad(){};
    //  copy, but change the flags
    Shape(Shape const &ref, hnnx::ShapeFlag newflags) : Shape(ref) { flags = unsigned(newflags); }

    std::array<size_t, Rank> dims;
    std::array<size_t, Rank> max_dims;
    std::array<uint8_t, Rank> pad;
    static constexpr size_t RankVal = Rank;
    // make crated shape matching given shape, or re-use an existing crated shape
    static const Shape *canonical_shape(Graph &graph_in, const Shape &val);
    static const Shape *canonical_shape(Graph &graph_in, const OutputDef &def);
    // force a given ShapeFlag state
    static const Shape *canonical_shape(Graph &graph_in, const Shape &val, hnnx::ShapeFlag newflags);
    // copy into crate without checking for existing duplicate
    static const Shape *crated_shape(Graph &graph_in, const Shape &val);

    bool operator<(const Shape &rhs) const { return std::memcmp(this, &rhs, shplen()) < 0; }
    static const Shape *deserialize(hnnx::Deserializer &dctx);
    void serialize(hnnx::Serializer &sctx) const;

  protected:
    unsigned shplen() const { return (char const *)&pad[0] + Rank - (char const *)this; }
};
// FIXME: this is incomplete since it doesn't have Shape<Rank> methods
// This doesn't have flags either, so there can be only one distinct instance of Shape<0>.
//
template <> struct Shape<0> {
    std::array<uint8_t, 1> dims;
    std::array<uint8_t, 1> max_dims;
    std::array<uint8_t, 1> pad;

  protected:
    unsigned shplen() const { return 0; }
};

#pragma GCC visibility pop

using Shapes = hnnx::shape_reduce_map[7];

#if 0
struct ShapeRepository {
    Shapes shapes;
};
#endif

#endif
