//==============================================================================
//
// Copyright (c) 2020-2023 Qualcomm Technologies, Inc.
// All Rights Reserved.
// Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#ifndef OP_PACKAGE_OPS_OPTS_LIST_DEFS_H
#define OP_PACKAGE_OPS_OPTS_LIST_DEFS_H 1

#include "unique_types.h"
#include "c_tricks.h"

namespace hnnx {

/**
 * @brief process_op_registration_list
 * Register all of the Ops in the Op registration list
 */
void process_op_registration_list();

/**
 * @brief process_opt_registration_list
 * Register all of the Opts in the Opt registration list
 * 
 */
void process_opt_registration_list();

} // namespace hnnx

namespace fold {
template <auto, int> struct ModifiedDerivedType;
} //namespace fold

/** @brief IMPL_APPEND_REG_OP_ELEM_NO_TCM_FOLDING (used by REGISTER_OP, REGISTER_OP_HVX, etc.) */
#define IMPL_APPEND_REG_OP_ELEM_NO_TCM_FOLDING(I, FP, OP, TAG, IS_SIMPLE)                                              \
    /** @brief Increment the Op count for this file @return 1 */                                                       \
    template <> constexpr int32_t NC<(I)>::inc_op() noexcept { return 1; }                                             \
                                                                                                                       \
    /** @brief Whether the name for this Op is already present in the Op name string table */                          \
    template <>                                                                                                        \
    constexpr bool StrUpd<(I)>::is_new_op_name =                                                                       \
            make_string_view(op_name_chain<UniqTy<0>, I - 1>().get_arr()).find(std::string_view{(OP), sizeof(OP)}) ==  \
            std::string_view::npos;                                                                                    \
                                                                                                                       \
    /** @brief Whether the name for this type suffix is already present in the type suffix string table */             \
    template <>                                                                                                        \
    constexpr bool StrUpd<(I)>::is_new_type_suffix =                                                                   \
            make_string_view(type_tag_chain<UniqTy<0>, I - 1>().get_arr())                                             \
                    .find(std::string_view{(TAG).data(), (TAG).size()}) == std::string_view::npos;                     \
                                                                                                                       \
    /** @brief Update the size of the Op name string table for this file. @return 0 or sizeof(OP) */                   \
    template <> constexpr uint64_t NC<(I)>::inc_op_name_strtab_size() noexcept                                         \
    {                                                                                                                  \
        if (StrUpd<(I)>::is_new_op_name) {                                                                             \
            return sizeof(OP);                                                                                         \
        } else {                                                                                                       \
            return 0U;                                                                                                 \
        }                                                                                                              \
    }                                                                                                                  \
                                                                                                                       \
    /** @brief Update the size of the type suffix string table for this file. @return 0 or TAG.size() */               \
    template <> constexpr uint64_t NC<(I)>::inc_type_suffix_strtab_size() noexcept                                     \
    {                                                                                                                  \
        if (StrUpd<(I)>::is_new_type_suffix) {                                                                         \
            return (TAG).size();                                                                                       \
        } else {                                                                                                       \
            return 0U;                                                                                                 \
        }                                                                                                              \
    }                                                                                                                  \
                                                                                                                       \
    /** @brief Grow the Op name string table for this file. No-op if it already contains the string */                 \
    template <>                                                                                                        \
    template <>                                                                                                        \
    constexpr ba_str<NC<(I)>::op_name_strtab_size()>                                                                   \
            op_name_strtab_container::chain<UniqTy<0>, NC<(I)>::reg_op_count(), NC<(I)>::op_name_strtab_size()> =      \
                    strtab_append<UniqTy<0>, I, sizeof(OP), StrUpd<(I)>::is_new_op_name>(                              \
                            op_name_chain<UniqTy<0>, I - 1>(), std::string_view{(OP), sizeof(OP)});                    \
                                                                                                                       \
    /** @brief Get the offset of this Op name in the Op name string table. */                                          \
    template <>                                                                                                        \
    constexpr uint16_t StrUpd<(I)>::op_name_offset = static_cast<uint16_t>(                                            \
            make_string_view(op_name_chain<UniqTy<0>, I>().get_arr()).find(std::string_view{(OP), sizeof(OP)}));       \
                                                                                                                       \
    /** @brief Grow the type suffix string table for this file. No-op if it already contains the string */             \
    template <>                                                                                                        \
    template <>                                                                                                        \
    constexpr ba_str<NC<(I)>::type_suffix_strtab_size()> type_suffix_strtab_container::chain<                          \
            UniqTy<0>, NC<(I)>::reg_op_count(), NC<(I)>::type_suffix_strtab_size()> =                                  \
            strtab_append<UniqTy<0>, I, (TAG).size(), StrUpd<(I)>::is_new_type_suffix>(                                \
                    type_tag_chain<UniqTy<0>, I - 1>(), std::string_view{(TAG).data(), (TAG).size()});                 \
                                                                                                                       \
    /** @brief Record the offset of this type suffix in the type suffix string table. */                               \
    template <>                                                                                                        \
    constexpr uint16_t StrUpd<(I)>::type_suffix_offset =                                                               \
            static_cast<uint16_t>(make_string_view(type_tag_chain<UniqTy<0>, I>().get_arr())                           \
                                          .find(std::string_view{(TAG).data(), (TAG).size()}));                        \
                                                                                                                       \
    /** @brief Finally, append a new element to the Op registration table. */                                          \
    template <>                                                                                                        \
    template <>                                                                                                        \
    constexpr ba_op<NC<(I)>::reg_op_count()> op_arr_container::chain<UniqTy<0>, NC<(I)>::reg_op_count()> =             \
            chain<UniqTy<0>, NC<(I - 1)>::reg_op_count()>.append(                                                      \
                    hnnx::reg_op_node{hnnx::GetParms<IS_SIMPLE>::get<FP, I>(), StrUpd<(I)>::op_name_offset,            \
                                      StrUpd<(I)>::type_suffix_offset});

/** @brief IMPL_APPEND_REG_OP_ELEM (used by REGISTER_OP, REGISTER_OP_HVX, etc.) */
#define IMPL_APPEND_REG_OP_ELEM(I, FP, OP, TAG, LINE)                                                                  \
    /** @brief Increment the Op count for this file @return 1 */                                                       \
    template <> constexpr int32_t NC<(I)>::inc_op() noexcept { return 1; }                                             \
                                                                                                                       \
    /** @brief Whether the name for this Op is already present in the Op name string table */                          \
    template <>                                                                                                        \
    constexpr bool StrUpd<(I)>::is_new_op_name =                                                                       \
            make_string_view(op_name_chain<UniqTy<0>, I - 1>().get_arr()).find(std::string_view{(OP), sizeof(OP)}) ==  \
            std::string_view::npos;                                                                                    \
                                                                                                                       \
    /** @brief Whether the name for this type suffix is already present in the type suffix string table */             \
    template <>                                                                                                        \
    constexpr bool StrUpd<(I)>::is_new_type_suffix =                                                                   \
            make_string_view(type_tag_chain<UniqTy<0>, I - 1>().get_arr())                                             \
                    .find(std::string_view{(TAG).data(), (TAG).size()}) == std::string_view::npos;                     \
                                                                                                                       \
    /** @brief Update the size of the Op name string table for this file. @return 0 or sizeof(OP) */                   \
    template <> constexpr uint64_t NC<(I)>::inc_op_name_strtab_size() noexcept                                         \
    {                                                                                                                  \
        if (StrUpd<(I)>::is_new_op_name) {                                                                             \
            return sizeof(OP);                                                                                         \
        } else {                                                                                                       \
            return 0U;                                                                                                 \
        }                                                                                                              \
    }                                                                                                                  \
                                                                                                                       \
    /** @brief Update the size of the type suffix string table for this file. @return 0 or TAG.size() */               \
    template <> constexpr uint64_t NC<(I)>::inc_type_suffix_strtab_size() noexcept                                     \
    {                                                                                                                  \
        if (StrUpd<(I)>::is_new_type_suffix) {                                                                         \
            return (TAG).size();                                                                                       \
        } else {                                                                                                       \
            return 0U;                                                                                                 \
        }                                                                                                              \
    }                                                                                                                  \
                                                                                                                       \
    /** @brief Grow the Op name string table for this file. No-op if it already contains the string */                 \
    template <>                                                                                                        \
    template <>                                                                                                        \
    constexpr ba_str<NC<(I)>::op_name_strtab_size()>                                                                   \
            op_name_strtab_container::chain<UniqTy<0>, NC<(I)>::reg_op_count(), NC<(I)>::op_name_strtab_size()> =      \
                    strtab_append<UniqTy<0>, I, sizeof(OP), StrUpd<(I)>::is_new_op_name>(                              \
                            op_name_chain<UniqTy<0>, I - 1>(), std::string_view{(OP), sizeof(OP)});                    \
                                                                                                                       \
    /** @brief Get the offset of this Op name in the Op name string table. */                                          \
    template <>                                                                                                        \
    constexpr uint16_t StrUpd<(I)>::op_name_offset = static_cast<uint16_t>(                                            \
            make_string_view(op_name_chain<UniqTy<0>, I>().get_arr()).find(std::string_view{(OP), sizeof(OP)}));       \
                                                                                                                       \
    /** @brief Grow the type suffix string table for this file. No-op if it already contains the string */             \
    template <>                                                                                                        \
    template <>                                                                                                        \
    constexpr ba_str<NC<(I)>::type_suffix_strtab_size()> type_suffix_strtab_container::chain<                          \
            UniqTy<0>, NC<(I)>::reg_op_count(), NC<(I)>::type_suffix_strtab_size()> =                                  \
            strtab_append<UniqTy<0>, I, (TAG).size(), StrUpd<(I)>::is_new_type_suffix>(                                \
                    type_tag_chain<UniqTy<0>, I - 1>(), std::string_view{(TAG).data(), (TAG).size()});                 \
                                                                                                                       \
    /** @brief Record the offset of this type suffix in the type suffix string table. */                               \
    template <>                                                                                                        \
    constexpr uint16_t StrUpd<(I)>::type_suffix_offset =                                                               \
            static_cast<uint16_t>(make_string_view(type_tag_chain<UniqTy<0>, I>().get_arr())                           \
                                          .find(std::string_view{(TAG).data(), (TAG).size()}));                        \
                                                                                                                       \
    /** @brief Finally, append a new element to the Op registration table. */                                          \
    /** @brief IS_SIMPLE argument to GetParms::get is always false; we only fold for internal ops, not op packages */  \
    template <>                                                                                                        \
    template <>                                                                                                        \
    constexpr ba_op<NC<(I)>::reg_op_count()> op_arr_container::chain<UniqTy<0>, NC<(I)>::reg_op_count()> =             \
            chain<UniqTy<0>, NC<(I - 1)>::reg_op_count()>.append(                                                      \
                    hnnx::reg_op_node{hnnx::GetParms<false>::get<fold::ModifiedDerivedType<FP, LINE>::Modified, I>(),  \
                                      StrUpd<(I)>::op_name_offset, StrUpd<(I)>::type_suffix_offset});

/** @brief APPEND_REG_OP_ELEM (used by REGISTER_OP, REGISTER_OP_HVX, etc.) */
#define APPEND_REG_OP_ELEM(FP, OP, TAG, LINE) IMPL_APPEND_REG_OP_ELEM(__COUNTER__, FP, OP, TAG, LINE)
/** @breif see register-op-tcm-folding.md **/
#define APPEND_REG_OP_ELEM_NO_TCM_FOLDING(FP, OP, TAG, IS_SIMPLE)                                                      \
    IMPL_APPEND_REG_OP_ELEM_NO_TCM_FOLDING(__COUNTER__, FP, OP, TAG, IS_SIMPLE)

/** @brief IMPL_APPEND_REG_OPT_ELEM (used by DEF_OPT and DEF_OPTIM) */
#define IMPL_APPEND_REG_OPT_ELEM(I, PRIORITY, FLAGS, DEFOPTFN, LINE)                                                   \
                                                                                                                       \
    /** @brief Increment the Optimization count for this file @return 1 */                                             \
    template <> constexpr int32_t NC<(I)>::inc_opt() noexcept { return 1; }                                            \
                                                                                                                       \
    /** @brief Append a new element to the Optimization registration table. */                                         \
    template <>                                                                                                        \
    template <>                                                                                                        \
    constexpr ba_opt<NC<(I)>::reg_opt_count()> opt_arr_container::chain<UniqTy<0>, NC<(I)>::reg_opt_count()> =         \
            chain<UniqTy<0>, NC<(I - 1)>::reg_opt_count()>.append(hnnx::reg_optim_node{                                \
                    static_cast<uint16_t>(PRIORITY), (FLAGS), (DEFOPTFN), static_cast<uint16_t>(LINE)});

/** @brief APPEND_REG_OPT_ELEM (used by DEF_OPT and DEF_OPTIM) */
#define APPEND_REG_OPT_ELEM(PRIORITY, FLAGS, DEFOPTFN, LINE)                                                           \
    IMPL_APPEND_REG_OPT_ELEM(__COUNTER__, PRIORITY, FLAGS, DEFOPTFN, LINE)

#define IMPL_INITIALIZE_TABLES(COUNT)                                                                                  \
    DEFINE_UNIQ_TY()                                                                                                   \
    using hnnx::reg_op_node;                                                                                           \
    using hnnx::reg_optim_node;                                                                                        \
    using hnnx::built_array;                                                                                           \
    using hnnx::ba_op;                                                                                                 \
    using hnnx::ba_opt;                                                                                                \
    using hnnx::ba_str;                                                                                                \
    using hnnx::NodeCounter;                                                                                           \
    using hnnx::op_arr_container;                                                                                      \
    using hnnx::opt_arr_container;                                                                                     \
    using hnnx::op_name_strtab_container;                                                                              \
    using hnnx::type_suffix_strtab_container;                                                                          \
    using hnnx::StrtabUpdate;                                                                                          \
    using hnnx::strtab_append;                                                                                         \
    using hnnx::make_string_view;                                                                                      \
    using hnnx::op_name_chain;                                                                                         \
    using hnnx::type_tag_chain;                                                                                        \
    namespace {                                                                                                        \
    template <int32_t I> using NC = NodeCounter<UniqTy<0>, I>;                                                         \
    template <int32_t I> using StrUpd = StrtabUpdate<UniqTy<0>, I>;                                                    \
    }                                                                                                                  \
    template <> constexpr int32_t NC<(COUNT)>::reg_op_count() noexcept { return 0; }                                   \
    template <> constexpr int32_t NC<(COUNT)>::reg_opt_count() noexcept { return 0; }                                  \
    template <> constexpr uint64_t NC<(COUNT)>::op_name_strtab_size() noexcept { return 0U; }                          \
    template <> constexpr uint64_t NC<(COUNT)>::type_suffix_strtab_size() noexcept { return 0U; }                      \
    template <> template <> constexpr ba_op<0> op_arr_container::chain<UniqTy<0>, 0> = {};                             \
    template <> template <> constexpr ba_opt<0> opt_arr_container::chain<UniqTy<0>, 0> = {};                           \
    template <> template <> constexpr ba_str<0> op_name_strtab_container::chain<UniqTy<0>, 0, 0> = {};                 \
    template <> template <> constexpr ba_str<0> type_suffix_strtab_container::chain<UniqTy<0>, 0, 0> = {};

#define INITIALIZE_TABLES() IMPL_INITIALIZE_TABLES(__COUNTER__)

#define OPS_REG_TABLE(NAME)         CTRICKS_PASTER(NAME, _inner_ops_regist_table)
#define OP_NAME_STR_TABLE(NAME)     CTRICKS_PASTER(NAME, _inner_op_name_strtab)
#define TYPE_SUFFIX_STR_TABLE(NAME) CTRICKS_PASTER(NAME, _inner_type_suffix_strtab)
#define EXT_OPS_REG_TABLE(NAME)     CTRICKS_PASTER(NAME, _ops_table)

#define OPTS_REG_TABLE(NAME)     CTRICKS_PASTER(NAME, _inner_opts_regist_table)
#define EXT_OPTS_REG_TABLE(NAME) CTRICKS_PASTER(NAME, _opts_table)

/**
 * @brief IMPL_FINALIZE_TABLES defines the registration tables for both 
 * the ops and opts defined in the Op source file
 */
#define IMPL_FINALIZE_TABLES(COUNT, NAME)                                                                                                                                   \
    namespace {                                                                                                                                                             \
    /** @brief The completed Op registration table */                                                                                                                       \
    auto const OPS_REG_TABLE(NAME) = op_arr_container::chain<UniqTy<0>, NC<(COUNT)>::reg_op_count()>.get_arr();                                                             \
    /** @brief The completed Op name string table */                                                                                                                        \
    auto const OP_NAME_STR_TABLE(NAME) =                                                                                                                                    \
            op_name_strtab_container::chain<UniqTy<0>, NC<(COUNT)>::reg_op_count(), NC<(COUNT)>::op_name_strtab_size()>.get_arr();                                          \
    /** @brief The completed type suffix string table */                                                                                                                    \
    auto const TYPE_SUFFIX_STR_TABLE(NAME) = type_suffix_strtab_container::chain<UniqTy<0>, NC<(COUNT)>::reg_op_count(), NC<(COUNT)>::type_suffix_strtab_size()>.get_arr(); \
    /** @brief The completed Optimization registration table */                                                                                                             \
    auto const OPTS_REG_TABLE(NAME) = opt_arr_container::chain<UniqTy<0>, NC<(COUNT)>::reg_opt_count()>.get_arr();                                                          \
    }                                                                                                                                                                       \
    namespace hnnx {                                                                                                                                                        \
    /** @brief Exported reference to the Op registration table, its associated string tables, and their sizes */                                                            \
    reg_op_table EXT_OPS_REG_TABLE(NAME){::OPS_REG_TABLE(NAME).data(),       ::OPS_REG_TABLE(NAME).size(),                                                                  \
                                         OP_NAME_STR_TABLE(NAME).data(),     OP_NAME_STR_TABLE(NAME).size(),                                                                \
                                         TYPE_SUFFIX_STR_TABLE(NAME).data(), TYPE_SUFFIX_STR_TABLE(NAME).size()};                                                           \
    /** @brief Exported reference to the Optimization registration table and its size */                                                                                    \
    reg_opt_table EXT_OPTS_REG_TABLE(NAME){::OPTS_REG_TABLE(NAME).data(), ::OPTS_REG_TABLE(NAME).size(), __FILE__};                                                         \
    }

/**
 * @brief FINALIZE_TABLES is a thunk to IMPL_FINALIZE_TABLES
 * 
 */
#define FINALIZE_TABLES(NAME) IMPL_FINALIZE_TABLES(__COUNTER__, NAME)

// The following macros are applied in ops_opts_registration.cc

/**
 * The "alias" attribute needs the mangled name of the variable to work correctly.
 * This name differs on GNU and Windows builds.
 */
#ifndef _MSC_VER
#define EMPTY_OPS_TABLE  _ZN4hnnx23default_empty_ops_tableE
#define EMPTY_OPTS_TABLE _ZN4hnnx24default_empty_opts_tableE
#else
#include "windows_op_registration.h"
#endif

#if defined(_MSC_VER) && !defined(__clang__)
// This is MSVC cl.exe, not clang-cl, so use the /alternatename pragma
// instead of __attribute((weak))__

#define OPS_TABLE_WEAK_SYMBOL(NAME)                                                                                    \
    MSVC_LINKER_PRAGMA(MANGLE_WIN_OPS_TABLE(EXT_OPS_REG_TABLE(NAME)), EMPTY_OPS_TABLE)                                 \
    extern reg_op_table EXT_OPS_REG_TABLE(NAME);

#define OPTS_TABLE_WEAK_SYMBOL(NAME)                                                                                   \
    MSVC_LINKER_PRAGMA(MANGLE_WIN_OPTS_TABLE(EXT_OPTS_REG_TABLE(NAME)), EMPTY_OPTS_TABLE)                              \
    extern reg_opt_table EXT_OPTS_REG_TABLE(NAME);

#else

#define DECLARE_CLANG_WEAK_SYMBOL2(TYPE, IDENT, ALIAS) extern TYPE IDENT __attribute__((weak, alias(#ALIAS)));

#define DECLARE_CLANG_WEAK_SYMBOL(TYPE, ID, ALIAS) DECLARE_CLANG_WEAK_SYMBOL2(TYPE, ID, ALIAS)

#define OPS_TABLE_WEAK_SYMBOL(NAME) DECLARE_CLANG_WEAK_SYMBOL(reg_op_table, EXT_OPS_REG_TABLE(NAME), EMPTY_OPS_TABLE)
#define OPTS_TABLE_WEAK_SYMBOL(NAME)                                                                                   \
    DECLARE_CLANG_WEAK_SYMBOL(reg_opt_table, EXT_OPTS_REG_TABLE(NAME), EMPTY_OPTS_TABLE)

#endif

/**
 * @brief As part of loading the HTP core, register all of the Ops and Optimization rules.                              
 * This will be called at static-initialization time.                                                                   
*/
#define OP_OPT_PROCESSOR(PREFIX)                                                                                       \
    namespace hnnx {                                                                                                   \
    void PREFIX##process_op_registration_list()                                                                        \
    {                                                                                                                  \
        const uint32_t size = ::PREFIX##_op_package_ops_list.size();                                                   \
        for (uint32_t i = 0U; i < size; i++) {                                                                         \
            reg_op_table const *const op_tab = ::PREFIX##_op_package_ops_list[i].get_ptr();                            \
            reg_op_node const *const entries = op_tab->get_entries();                                                  \
            std::string_view const names = op_tab->get_op_name_strtab();                                               \
            std::string_view const suffixes = op_tab->get_type_suffix_strtab();                                        \
            for (uint32_t j = 0U; j < op_tab->get_num_entries(); j++) {                                                \
                const reg_op_node reg_op = entries[j];                                                                 \
                reg_op.PREFIX##_process(names, suffixes);                                                              \
            }                                                                                                          \
        }                                                                                                              \
    }                                                                                                                  \
    void PREFIX##process_opt_registration_list()                                                                       \
    {                                                                                                                  \
        const uint32_t size = ::PREFIX##_op_package_opts_list.size();                                                  \
        for (uint32_t i = 0U; i < size; i++) {                                                                         \
            reg_opt_table const *const opt_tab = ::PREFIX##_op_package_opts_list[i].get_ptr();                         \
            reg_optim_node const *const entries = opt_tab->get_entries();                                              \
            std::string_view const fname = opt_tab->get_file_name();                                                   \
            for (uint32_t j = 0U; j < opt_tab->get_num_entries(); j++) {                                               \
                const reg_optim_node reg_opt = entries[j];                                                             \
                reg_opt.PREFIX##_process(fname);                                                                       \
            }                                                                                                          \
        }                                                                                                              \
    }                                                                                                                  \
    static void PREFIX##ops_opts_registration()                                                                        \
    {                                                                                                                  \
        PREFIX##process_op_registration_list();                                                                        \
        PREFIX##process_opt_registration_list();                                                                       \
    }                                                                                                                  \
    /** Force Ops and Opts to be registered at static-init time*/                                                      \
    /** This avoids init-time regressions when Graph::init_once() is called during graph creation.*/                   \
    [[maybe_unused]] static bool PREFIX##REGISTER_OPS_AND_OPTS = (PREFIX##ops_opts_registration(), true);              \
    } // namespace hnnx

#define IMPL_BEGIN_OPS_OPTS_LIST(I)                                                                                    \
    namespace hnnx {                                                                                                   \
    [[maybe_unused]] reg_op_table default_empty_ops_table{};                                                           \
    [[maybe_unused]] reg_opt_table default_empty_opts_table{};                                                         \
    template <> template <> constexpr ba_op_table<0> op_table_arr_container::chain<UniqTy<0>, (I)> = {};               \
    template <> template <> constexpr ba_opt_table<0> opt_table_arr_container::chain<UniqTy<0>, (I)> = {};             \
    }

/** @brief Begin defining the list of Ops/Opts registration lists */
#define BEGIN_OPS_OPTS_LIST()     IMPL_BEGIN_OPS_OPTS_LIST(__COUNTER__)
#define BEGIN_PKG_OPS_OPTS_LIST() IMPL_BEGIN_OPS_OPTS_LIST(__COUNTER__)

#define IMPL_END_OPS_OPTS_LIST(PREFIX, COUNT)                                                                          \
    /** Append the empty 'default' tables here, so we don't violate AUTOSAR by leaving them unused. */                 \
    namespace hnnx {                                                                                                   \
    template <>                                                                                                        \
    template <>                                                                                                        \
    constexpr ba_op_table<(COUNT)> op_table_arr_container::chain<UniqTy<0>, (COUNT)> =                                 \
            chain<UniqTy<0>, (COUNT)-1>.append(&default_empty_ops_table);                                              \
                                                                                                                       \
    template <>                                                                                                        \
    template <>                                                                                                        \
    constexpr ba_opt_table<(COUNT)> opt_table_arr_container::chain<UniqTy<0>, (COUNT)> =                               \
            chain<UniqTy<0>, (COUNT)-1>.append(&default_empty_opts_table);                                             \
    }                                                                                                                  \
                                                                                                                       \
    namespace {                                                                                                        \
    /** @brief PREFIX_op_package_ops_list references all of the registered ops for the HTP Core. */                    \
    auto const PREFIX##_op_package_ops_list = hnnx::op_table_arr_container::chain<UniqTy<0>, (COUNT)>.get_arr();       \
                                                                                                                       \
    /** @brief PREFIX_op_package_opts_list references all of the registered graph optimizations for the HTP Core. */   \
    auto const PREFIX##_op_package_opts_list = hnnx::opt_table_arr_container::chain<UniqTy<0>, (COUNT)>.get_arr();     \
    }

/** @brief Finish defining the list of Ops/Opts registration lists */
#define END_OPS_OPTS_LIST() IMPL_END_OPS_OPTS_LIST(core, __COUNTER__) OP_OPT_PROCESSOR(core)

#define END_PKG_OPS_OPTS_LIST() IMPL_END_OPS_OPTS_LIST(pkg, __COUNTER__) OP_OPT_PROCESSOR(pkg)

/** @brief Declare the list of registered ops and optimizations
 *  for the given op file, and append it to the linked list.
 *
 *  A weak definition is provided for the registration list that initializes it to an 'empty' list
 *  (only the null sentinel node is present). Op source files will then override these weak definitions
 *  with strong ones containing the actual list.
 */
#define IMPL_DECLARE_OPS_OPTS_LIST(I, NAME)                                                                            \
    namespace hnnx {                                                                                                   \
    OPS_TABLE_WEAK_SYMBOL(NAME)                                                                                        \
                                                                                                                       \
    template <>                                                                                                        \
    template <>                                                                                                        \
    constexpr ba_op_table<(I)>                                                                                         \
            op_table_arr_container::chain<UniqTy<0>, (I)> = chain<UniqTy<0>, (I)-1>.append(&EXT_OPS_REG_TABLE(NAME));  \
                                                                                                                       \
    OPTS_TABLE_WEAK_SYMBOL(NAME)                                                                                       \
                                                                                                                       \
    template <>                                                                                                        \
    template <>                                                                                                        \
    constexpr ba_opt_table<(I)> opt_table_arr_container::chain<UniqTy<0>, (I)> =                                       \
            chain<UniqTy<0>, (I)-1>.append(&EXT_OPTS_REG_TABLE(NAME));                                                 \
    }

#define DECLARE_OPS_OPTS_LIST(NAME)     IMPL_DECLARE_OPS_OPTS_LIST(__COUNTER__, NAME)
#define DECLARE_PKG_OPS_OPTS_LIST(NAME) IMPL_DECLARE_OPS_OPTS_LIST(__COUNTER__, NAME)

#endif
