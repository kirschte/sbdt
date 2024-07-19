#ifndef TREENODE_H
#define TREENODE_H

#include <vector>


class TreeNode {
public:
    // constructors
    TreeNode(bool is_leaf);
    ~TreeNode();

    // fields
    TreeNode *left, *right;
    int depth;
    int split_attr;
    int split_index;
    double split_value;
    double split_gain;
    double split_gain_orig;
    double split_gain_top;
    int lhs_size, rhs_size;
    double prediction; // if it's a leaf
    int n;
    std::vector<int> identifiers = std::vector<int>();

    // methods
    bool is_leaf() const;
};

inline bool TreeNode::is_leaf() const {
    return (left == nullptr && right == nullptr);
}


#endif // TREENODE_H