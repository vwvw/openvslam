#include "openvslam/data/frame.h"
#include "openvslam/data/keyframe.h"
#include "openvslam/data/frame_statistics.h"

namespace openvslam {
namespace data {

void frame_statistics::update_frame_statistics(const data::frame& frm, const bool is_lost) {
    if (frm.cam_pose_cw_is_valid_) {
        const Mat44_t rel_cam_pose_from_ref_keyfrm = frm.cam_pose_cw_ * frm.ref_keyfrm_->get_cam_pose_inv();

        frm_ids_of_ref_keyfrms_[frm.ref_keyfrm_].push_back(frm.id_);

        ++num_valid_frms_;
        assert(!ref_keyfrms_.count(frm.id_));
        ref_keyfrms_[frm.id_] = frm.ref_keyfrm_;
        assert(!rel_cam_poses_from_ref_keyfrms_.count(frm.id_));
        rel_cam_poses_from_ref_keyfrms_[frm.id_] = rel_cam_pose_from_ref_keyfrm;
        assert(!timestamps_.count(frm.id_));
        timestamps_[frm.id_] = frm.timestamp_;
    }

    assert(!is_lost_frms_.count(frm.id_));
    is_lost_frms_[frm.id_] = is_lost;
}

void frame_statistics::replace_reference_keyframe(const std::shared_ptr<data::keyframe>& old_keyfrm, const std::shared_ptr<data::keyframe>& new_keyfrm) {
    // keyframeを削除する時は以下の手順で対応関係の更新を行う
    // 1. frm_ids_of_ref_keyfrms_で削除対象のkeyframeを参照しているframeのIDsを検索する
    // 2. ref_keyfrms_.at(ID)を削除対象のkeyframeのparentに変更する
    // 3. rel_cam_poses_from_ref_keyfrms_.at(ID)を削除対象のkeyframeのparentまでの相対姿勢に変更
    // 4. frm_ids_of_ref_keyfrms_のkeyを削除対象のkeyframeからparentに変更

    assert(num_valid_frms_ == rel_cam_poses_from_ref_keyfrms_.size());
    assert(num_valid_frms_ == ref_keyfrms_.size());
    assert(num_valid_frms_ == timestamps_.size());
    assert(num_valid_frms_ <= is_lost_frms_.size());

    // キーフレームを置き換える必要がなければ終了
    if (!frm_ids_of_ref_keyfrms_.count(old_keyfrm)) {
        return;
    }

    // old_keyfrmを参照しているframeを検索
    const auto frm_ids = frm_ids_of_ref_keyfrms_.at(old_keyfrm);

    for (const auto frm_id : frm_ids) {
        assert(*ref_keyfrms_.at(frm_id) == *old_keyfrm);

        // 古い方のキーフレームの姿勢・相対姿勢を取得しておく
        const Mat44_t old_ref_cam_pose_cw = old_keyfrm->get_cam_pose();
        const Mat44_t old_rel_cam_pose_cr = rel_cam_poses_from_ref_keyfrms_.at(frm_id);

        // キーフレームのポインタを置き換え
        ref_keyfrms_.at(frm_id) = new_keyfrm;

        // 相対姿勢を更新
        const Mat44_t new_ref_cam_pose_cw = new_keyfrm->get_cam_pose();
        const Mat44_t new_rel_cam_pose_cr = old_rel_cam_pose_cr * old_ref_cam_pose_cw * new_ref_cam_pose_cw.inverse();
        rel_cam_poses_from_ref_keyfrms_.at(frm_id) = new_rel_cam_pose_cr;
    }

    // mapのkeyを置き換え
    frm_ids_of_ref_keyfrms_[new_keyfrm] = std::move(frm_ids_of_ref_keyfrms_.at(old_keyfrm));
    frm_ids_of_ref_keyfrms_.erase(old_keyfrm);
}

std::unordered_map<std::shared_ptr<data::keyframe>, std::vector<unsigned int>> frame_statistics::get_frame_id_of_reference_keyframes() const {
    return frm_ids_of_ref_keyfrms_;
}

unsigned int frame_statistics::get_num_valid_frames() const {
    return num_valid_frms_;
}

std::map<unsigned int, std::shared_ptr<data::keyframe>> frame_statistics::get_reference_keyframes() const {
    return {ref_keyfrms_.begin(), ref_keyfrms_.end()};
}

eigen_alloc_map<unsigned int, Mat44_t> frame_statistics::get_relative_cam_poses() const {
    return {rel_cam_poses_from_ref_keyfrms_.begin(), rel_cam_poses_from_ref_keyfrms_.end()};
}

std::map<unsigned int, double> frame_statistics::get_timestamps() const {
    return {timestamps_.begin(), timestamps_.end()};
}

std::map<unsigned int, bool> frame_statistics::get_lost_frames() const {
    return {is_lost_frms_.begin(), is_lost_frms_.end()};
}

void frame_statistics::clear() {
    num_valid_frms_ = 0;
    frm_ids_of_ref_keyfrms_.clear();
    ref_keyfrms_.clear();
    rel_cam_poses_from_ref_keyfrms_.clear();
    timestamps_.clear();
    is_lost_frms_.clear();
}

} // namespace data
} // namespace openvslam
