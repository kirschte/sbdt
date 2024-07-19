#include "tree_node.h"


TreeNode::TreeNode(bool is_leaf): depth(0), split_attr(-1), split_value(-1), split_gain(-1), split_gain_orig(-1), split_gain_top(-1), lhs_size(0), rhs_size(0), prediction(0), n(0)
{
    if (is_leaf) {
        left = nullptr; right = nullptr;
    } else {
        left = this; right = this;
    }
}

TreeNode::~TreeNode() {}
