#include <stdlib.h>
#include <cstdint>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <xmmintrin.h>
#include <immintrin.h>
#include <emmintrin.h>
#include <smmintrin.h>
#include <bit>
#include <functional>

#define ARR_SIZE 4
#define MAXHEIGHT 50

#define PREFETCH_DISTANCE 8

typedef uint64_t Key; // Key is an 8-byte integer

typedef std::chrono::high_resolution_clock Clock;

int compare_(const Key& a, const Key& b) {
    if (a < b) {
        return -1;
    } else if (a > b) {
        return +1;
    } else {
        return 0;
    }
}

template<typename Key>
class SkipList {
   private:
    struct Node;

   public:
    SkipList();

    // DASL's insertion functions
    void Insert_usplit(const Key& key); // Code for insertion with uneven-split
    void Insert_esplit(const Key& key); // Code for insertion with even-split
    
    void Insert_Raise(const Key& key);
    void Insert_Search(const Key& key);
    void Insert_Array(const Key& key);

    void Insert_future(const Key& key); // Don't mind this function

    // DASL's lookup functions
    bool Contains(const Key& key) const; 
    bool Contains_Height(const Key& key) const;

    // DASL's Scan functions
    Key Scan(const Key& key, const int scan_num);

    // DASL's profiling functions
    void Array_utilization();
    void Print() const;
    inline int print_shift() { return shift_count; }
    inline int print_split_cnt() { return split_count; }
    inline int print_height() { return max_height_; }

   private:
    int kMaxHeight_;
    Node* head_[MAXHEIGHT];
    int max_height_;

    int shift_count;
    int split_count;

    inline int GetMaxHeight() const {
        return max_height_;
    }

    inline int randomLevel() {
        int level = 0;
        while (std::rand() % 4 == 0 && level < MAXHEIGHT) { level++; }
        if (level+1 > max_height_) { max_height_ = level+1; }
        return level;
    }

    Node* NewNode(const Key& key); // Make a new node with key

    // Intra-node search with linear search and SIMD
    int findMaxLessOrEqualLinear(Key arr[], Key target, int size) const;
    int findMaxLessOrEqualLinearSIMD(Key arr[], Key target, int size) const;

    // Intra-node search with branchless binary search
    // findMaxLessOrEqual == findMaxLessOrEqualBinary
    int findMaxLessOrEqual(Key arr[], Key target) const; 
    int findMaxLessOrEqualBinary(Key arr[], Key target) const; 
};


template<typename Key>
struct SkipList<Key>::Node {
    Key keys[ARR_SIZE]; // keys[0] = leader key of current node
    Node* forward;
    Node* next[ARR_SIZE];
    int N_key;

    Node(Key key) : N_key(1) {
        this->forward = nullptr;
        for(int i = 0; i < ARR_SIZE; i++) {
            keys[i] = 0;
            next[i] = nullptr;
        }
        this->keys[0] = key;
    }
};

template<typename Key>
typename SkipList<Key>::Node*
SkipList<Key>::NewNode(const Key& key) {
    return new Node(key);
}

template<typename Key>
SkipList<Key>::SkipList() {
    kMaxHeight_ = MAXHEIGHT;
    max_height_ = 1;
    shift_count = 0;
    split_count = 0;

    for(int i = 0; i < kMaxHeight_; i++) {
        head_[i] = NewNode(Key());  
    }
}

template<typename Key>
int SkipList<Key>::findMaxLessOrEqualLinear(Key arr[], Key target, int size) const {
    for (int i = 0; i < size; ++i) {
        if (compare_(arr[i], target) > 0) {
            return i - 1;
        }
    }
    return size - 1;
}

template<typename Key>
int SkipList<Key>::findMaxLessOrEqualLinearSIMD(Key arr[], Key target, int size) const {
    constexpr int simdWidth = 2;
    __m128i targetVec = _mm_set1_epi64x(static_cast<uint64_t>(target));
    int maxIndex = -1;
    int i = 0;
    for (; i <= size - simdWidth; i += simdWidth) {
        __m128i dataVec = _mm_loadu_si128(reinterpret_cast<const __m128i*>(&arr[i]));
        __m128i cmp_gt = _mm_cmpgt_epi64(dataVec, targetVec);
        __m128i cmp_le = _mm_cmpeq_epi64(cmp_gt, _mm_setzero_si128());
        int mask = _mm_movemask_pd(_mm_castsi128_pd(cmp_le));

        if (mask != 0) {
            int bitPos = (mask == 1) ? 0 : 1;
            maxIndex = i + bitPos;
        }
    }
    for (; i < size; ++i) {
        if (compare_(arr[i], target) <= 0) {
            maxIndex = i;
        }
    }
    return maxIndex;
}

template<typename Key>
int SkipList<Key>::findMaxLessOrEqualBinary(Key arr[], Key target) const {
    Key* begin = arr;
    std::size_t step = ARR_SIZE;
    
    for (step /= 2; step != 0; step /= 2) {
        Key* probe = begin + step;
        bool valid = (*probe <= target) && *probe != 0;
        begin += step * valid;
    }
    return begin - arr;
}

template<typename Key>
int SkipList<Key>::findMaxLessOrEqual(Key arr[], Key target) const {
    Key* begin = arr;
    std::size_t step = ARR_SIZE;
    
    for (step /= 2; step != 0; step /= 2) {
        Key* probe = begin + step;
        bool valid = (*probe <= target) && *probe != 0;
        begin += step * valid;
    }
    return begin - arr;
}

template<typename Key>
void SkipList<Key>::Insert_usplit(const Key& key) {
    Node* prev_[MAXHEIGHT];
    std::copy(std::begin(head_), std::end(head_), std::begin(prev_));
    int height = GetMaxHeight() - 1; // Using for search
    Node* x = head_[height]; // Use when searching
    
    if (x->forward != nullptr && compare_(x->forward->keys[0], key) <= 0) x = x->forward;

    while (true) { // Find the location which will insert the key using prev_ and head_
        prev_[height--] = x;
        if (height >= 0) {
            int n_key = x->N_key;
            if (n_key <= ARR_SIZE/2) {
                x = (x == head_[height + 1]) ? head_[height] : x->next[findMaxLessOrEqualLinearSIMD(x->keys, key, n_key)];
            } else {
                x = (x == head_[height + 1]) ? head_[height] : x->next[findMaxLessOrEqualBinary(x->keys, key)];
            }
        } else {
            break;
        }
    }
    // Do not allow duplicated key
    if (prev_[0] != head_[0]) {
        for (int i = 0; i < prev_[0]->N_key; i++) {
            if (compare_(prev_[0]->keys[i], key) == 0) {
                return;
            }
        }
    }
    
    // Do insert operation
    int level = 0;
    while (true) {
        int stop_flag = 0;
        int cur_height = GetMaxHeight() - 1;
        if (prev_[level] == head_[level] && prev_[level]->forward == nullptr) {
        // Case 1: There is no node in list and need to make a new node
            if (level == 0) { // Case 1-1: If insert into H0
                Node* Elist_node = NewNode(key);
                Elist_node->forward = prev_[level]->forward;
                prev_[level]->forward = Elist_node;
                break;
            } else { // Case 1-2: If not insert into H0
                if (cur_height < level) {
                    max_height_++;
                }
                if (prev_[level-1]->forward != nullptr && prev_[level-1] == head_[level-1]) {
                    Node* Elist_node = NewNode(prev_[level-1]->forward->keys[0]);
                    Elist_node->forward = prev_[level]->forward;
                    Elist_node->next[0] = prev_[level-1]->forward;
                    prev_[level]->forward = Elist_node;
                } else {
                    Node* Elist_node = NewNode(prev_[level-1]->keys[0]);
                    Elist_node->forward = prev_[level]->forward;
                    Elist_node->next[0] = prev_[level-1];
                    prev_[level]->forward = Elist_node;
                }
                break;
            }
        } else if (prev_[level] == head_[level]) {
            // Case 2: There is another node in list and need to make a new node
            if (prev_[level]->forward->N_key != ARR_SIZE) { // Case 2-1: Forward node has a room
                if (level == 0) { // Case 2-1-1: Insert into H0
                    int idx = findMaxLessOrEqual(prev_[level]->forward->keys, key);
                    if (idx == 0 && prev_[level]->forward->keys[0] > key) idx = -1;

                    if (prev_[level]->forward->keys[idx] == key) {
                        break;
                    } else if (prev_[level]->forward->keys[idx+1] == 0) {
                        prev_[level]->forward->keys[idx+1] = key;
                        prev_[level]->forward->N_key++;
                        if (prev_[level]->forward->N_key == ARR_SIZE) {
                            level++;
                            if (cur_height < level) {
                                max_height_++;
                            } 
                        } else break;
                    } else {
                        Key update_key = prev_[level]->forward->keys[0];
                        std::memmove(&prev_[level]->forward->keys[idx+2], &prev_[level]->forward->keys[idx+1], (prev_[level]->forward->N_key - (idx+1)) * sizeof(Key));
                        shift_count++; // Signal.Jin
                        prev_[level]->forward->keys[idx+1] = key;
                        prev_[level]->forward->N_key++;
                        if (idx == -1) {
                            for (int i = level+1; i < GetMaxHeight(); i++) {
                                if (prev_[i] != nullptr) {
                                    int idx = findMaxLessOrEqual(prev_[i]->forward->keys, update_key);
                                    if (prev_[i]->forward->keys[idx] == update_key) {
                                        prev_[i]->forward->keys[idx] = key;
                                    }
                                    if (idx != 0) break;
                                }
                            }
                        }

                        if (prev_[level]->forward->N_key == ARR_SIZE) {
                            level++;
                            if (cur_height < level) {
                                max_height_++;
                            } 
                        } else break;
                    }
                } else { // Case 2-1-2: Not insert into H0
                    int idx = findMaxLessOrEqual(prev_[level]->forward->keys, prev_[level-1]->forward->keys[0]);
                    if (idx == 0 && prev_[level]->forward->keys[0] > prev_[level-1]->forward->keys[0]) idx = -1;

                    if (prev_[level]->forward->keys[idx] == prev_[level-1]->forward->keys[0]) {
                        break;
                    } else if (prev_[level]->forward->keys[idx+1] == 0) {
                        prev_[level]->forward->keys[idx+1] = prev_[level-1]->forward->keys[0];
                        prev_[level]->forward->next[idx+1] = prev_[level-1]->forward;
                        prev_[level]->forward->N_key++;
                        if (prev_[level]->forward->N_key == ARR_SIZE) {
                            level++;
                            if (cur_height < level) {
                                max_height_++;
                            } 
                        } else break;
                    } else {
                        Key update_key = prev_[level]->forward->keys[0];
                        std::memmove(&prev_[level]->forward->keys[idx+2], &prev_[level]->forward->keys[idx+1], (prev_[level]->forward->N_key - (idx+1)) * sizeof(Key));
                        std::memmove(&prev_[level]->forward->next[idx+2], &prev_[level]->forward->next[idx+1], (prev_[level]->forward->N_key - (idx+1)) * sizeof(Key));
                        shift_count++; // Signal.Jin
                        prev_[level]->forward->keys[idx+1] = prev_[level-1]->forward->keys[0];
                        prev_[level]->forward->next[idx+1] = prev_[level-1]->forward;
                        prev_[level]->forward->N_key++;
                        if (idx == -1) {
                            for (int i = level+1; i < GetMaxHeight(); i++) {
                                if (prev_[i] != nullptr) {
                                    int idx = findMaxLessOrEqual(prev_[i]->forward->keys, update_key);
                                    if (prev_[i]->forward->keys[idx] == update_key) {
                                        prev_[i]->forward->keys[idx] = prev_[level-1]->forward->keys[0];
                                    }
                                    if (idx != 0) break;
                                }
                            }
                        }

                        if (prev_[level]->forward->N_key == ARR_SIZE) {
                            level++;
                            if (cur_height < level) {
                                max_height_++;
                            } 
                        } else break;
                    }
                }
            } else { // Case 2-2: Forward node has no room, so we need to make a new node
                if (level == 0) { // Case 2-2-1: Insert into H0
                    int idx = findMaxLessOrEqual(prev_[level]->forward->keys, key);
                    if (idx == 0 && prev_[level]->forward->keys[0] > key) {
                        Node* add_node = NewNode(key);
                        add_node->forward = prev_[level]->forward;
                        prev_[level]->forward = add_node;    
                    } else {
                        Node* add_node = NewNode(key);
                        std::memcpy(add_node->keys+1, &prev_[level]->forward->keys[idx+1], (ARR_SIZE - (idx+1)) * sizeof(Key));
                        std::memset(&prev_[level]->forward->keys[idx+1], 0, (ARR_SIZE - (idx+1)) * sizeof(Key));
                        shift_count++; // Signal.Jin
                        add_node->N_key += ARR_SIZE - (idx+1);
                        add_node->forward = prev_[level]->forward->forward;
                        prev_[level]->forward->N_key -= ARR_SIZE - (idx+1);
                        prev_[level]->forward->forward = add_node;
                        prev_[level] = add_node;
                        level++; // Keep tracking
                        if (cur_height < level) {
                            max_height_++;
                        }
                    }
                } else { // Case 2-2-2: Not insert into H0
                    if (prev_[level-1] == head_[level-1]) {
                        Node* add_node = NewNode(prev_[level-1]->forward->keys[0]);
                        add_node->forward = prev_[level]->forward;
                        add_node->next[0] = prev_[level-1]->forward;
                        prev_[level]->forward = add_node;
                    } else {
                        Node* add_node = NewNode(prev_[level-1]->keys[0]);
                        add_node->forward = prev_[level]->forward;
                        add_node->next[0] = prev_[level-1];
                        prev_[level]->forward = add_node;
                    }
                    break;
                }
            }
        } else {
            // Case 3: New node must be inserted between nodes or into prev_ node (not head_)
            if (prev_[level]->N_key != ARR_SIZE) { // Case 3-1: prev_ node has a room, so we insert into that node
                if (level == 0) { // Case 3-1-1: Insert into H0
                    int idx = findMaxLessOrEqual(prev_[level]->keys, key);
                    if (idx == 0 && prev_[level]->keys[0] > key) idx = -1;
                    if (prev_[level]->keys[idx] == key) {
                        stop_flag++;
                    } else {
                        if (prev_[level]->keys[idx+1] == 0) {
                            prev_[level]->keys[idx+1] = key;
                            prev_[level]->N_key++;    
                        } else {
                            std::memmove(&prev_[level]->keys[idx+2], &prev_[level]->keys[idx+1], (prev_[level]->N_key - (idx+1)) * sizeof(Key));
                            shift_count++; // Signal.Jin
                            prev_[level]->keys[idx+1] = key;
                            prev_[level]->N_key++;
                        }
                        if (prev_[level]->N_key == ARR_SIZE && prev_[level+1] == head_[level+1]) {
                            level++;
                            if (cur_height < level) {
                                max_height_++;
                            }
                        } else stop_flag++;
                    }
                } else { // Case 3-1-2: Not insert into H0
                    int idx = findMaxLessOrEqual(prev_[level]->keys, prev_[level-1]->keys[0]);
                    if (idx == 0 && prev_[level]->keys[0] > prev_[level-1]->keys[0]) idx = -1;
                    if (prev_[level]->keys[idx] == prev_[level-1]->keys[0]) {
                        stop_flag++;
                    } else {
                        if (prev_[level]->keys[idx+1] == 0) {
                            prev_[level]->keys[idx+1] = prev_[level-1]->keys[0];
                            prev_[level]->next[idx+1] = prev_[level-1];
                            prev_[level]->N_key++;
                        } else {
                            std::memmove(&prev_[level]->keys[idx+2], &prev_[level]->keys[idx+1], (prev_[level]->N_key - (idx+1)) * sizeof(Key));
                            std::memmove(&prev_[level]->next[idx+2], &prev_[level]->next[idx+1], (prev_[level]->N_key - (idx+1)) * sizeof(Key));
                            shift_count++; // Signal.Jin
                            prev_[level]->keys[idx+1] = prev_[level-1]->keys[0];
                            prev_[level]->next[idx+1] = prev_[level-1];
                            prev_[level]->N_key++;
                        }
                    }
                    if (prev_[level]->N_key == ARR_SIZE && prev_[level+1] == head_[level+1]) {
                        level++;
                        if (cur_height < level) {
                            max_height_++;
                        }
                    } else stop_flag++;
                }
            } else { // Case 3-2: prev_ node has no room, so we need to make a new node
                // When creating a new node, change prev to the newly created node.
                // Uneven-split operation
                if (level == 0) { // Case 3-2-1: Insert into H0
                    split_count++; // Signal.Jin
                    int idx = findMaxLessOrEqual(prev_[level]->keys, key);
                    if (idx == 0 && prev_[level]->keys[0] > key) idx = -1;
                    { 
                        if (idx == ARR_SIZE-1) {
                            {
                                Node* add_node = NewNode(key);
                                add_node->forward = prev_[level]->forward;
                                prev_[level]->forward = add_node;
                                prev_[level] = add_node;
                                level++;
                                if (cur_height < level) {
                                    max_height_++;
                                }
                            }
                        } else {
                            if (idx < ARR_SIZE / 2) {
                                Node* add_node = NewNode(prev_[level]->keys[ARR_SIZE/2]);
                                std::memcpy(add_node->keys, &prev_[level]->keys[ARR_SIZE/2], (ARR_SIZE/2) * sizeof(Key));
                                std::memset(&prev_[level]->keys[ARR_SIZE/2], 0, (ARR_SIZE/2) * sizeof(Key));
                                shift_count++; // Signal.Jin
                                add_node->N_key = ARR_SIZE/2;
                                add_node->forward = prev_[level]->forward;
                                prev_[level]->forward = add_node;
                                if (idx == ARR_SIZE / 2 - 1) {
                                    prev_[level]->keys[idx + 1] = key;
                                    prev_[level]->N_key = ARR_SIZE / 2 + 1;
                                } else {
                                    Key update_key = prev_[level]->keys[0];
                                    prev_[level]->N_key = ARR_SIZE / 2;
                                    std::memmove(&prev_[level]->keys[idx+2], &prev_[level]->keys[idx+1], (prev_[level]->N_key - (idx+1)) * sizeof(Key));
                                    shift_count++; // Signal.Jin
                                    prev_[level]->keys[idx+1] = key;
                                    prev_[level]->N_key++;
                                    if (idx == -1) {
                                        for (int update = level+1; update < GetMaxHeight(); update++) {
                                            int idx = findMaxLessOrEqual(prev_[update]->forward->keys, update_key);
                                            if (prev_[update]->forward->keys[idx] == update_key) {
                                                prev_[update]->forward->keys[idx] = key;
                                            }
                                            if (idx != 0) break;
                                        }
                                        stop_flag++;
                                    }
                                }
                                prev_[level] = add_node;
                                level++; // Keep tracking
                                if (cur_height < level) {
                                    max_height_++;
                                }
                            } else {
                                prev_[level]->N_key = ARR_SIZE / 2;
                                idx = idx - ARR_SIZE / 2;
                                Node* add_node = NewNode(prev_[level]->keys[ARR_SIZE/2]);
                                std::memcpy(add_node->keys, &prev_[level]->keys[ARR_SIZE/2], (ARR_SIZE/2) * sizeof(Key));
                                std::memset(&prev_[level]->keys[ARR_SIZE/2], 0, (ARR_SIZE/2) * sizeof(Key));
                                shift_count++; // Signal.Jin
                                add_node->N_key = ARR_SIZE/2 + 1;
                                add_node->forward = prev_[level]->forward;
                                prev_[level]->forward = add_node;
                                prev_[level] = add_node;
                                std::memmove(&prev_[level]->keys[idx+2], &prev_[level]->keys[idx+1], (prev_[level]->N_key - (idx+1)) * sizeof(Key));
                                shift_count++; // Signal.Jin
                                prev_[level]->keys[idx+1] = key;
                                level++;
                                if (cur_height < level) {
                                    max_height_++;
                                }
                            }
                        }
                    }
                } else { // Case 3-2-2: Not insert into H0
                    split_count++; // Signal.Jin
                    int idx = findMaxLessOrEqual(prev_[level]->keys, prev_[level-1]->keys[0]);
                    if (idx == 0 && prev_[level]->keys[0] > prev_[level-1]->keys[0]) idx = -1;
                    {
                        if (idx == ARR_SIZE-1) {
                            {
                                Node* add_node = NewNode(prev_[level-1]->keys[0]);
                                add_node->forward = prev_[level]->forward;
                                add_node->next[0] = prev_[level-1];
                                prev_[level]->forward = add_node;
                                prev_[level] = add_node;
                                level++;
                                if (cur_height < level) {
                                    max_height_++;
                                }
                            }
                        } else {
                            if (idx < ARR_SIZE / 2) {
                                Node* add_node = NewNode(prev_[level]->keys[ARR_SIZE/2]);
                                std::memcpy(add_node->keys, &prev_[level]->keys[ARR_SIZE/2], (ARR_SIZE/2) * sizeof(Key));
                                std::memcpy(add_node->next, &prev_[level]->next[ARR_SIZE/2], (ARR_SIZE/2) * sizeof(prev_[level]->next[0]));
                                std::memset(&prev_[level]->keys[ARR_SIZE/2], 0, (ARR_SIZE/2) * sizeof(Key));
                                shift_count++; // Signal.Jin
                                for (int i = ARR_SIZE/2; i < ARR_SIZE; i++) {
                                    prev_[level]->next[i] = nullptr;
                                }
                                add_node->N_key = ARR_SIZE/2;
                                add_node->forward = prev_[level]->forward;
                                prev_[level]->forward = add_node;
                                if (prev_[level]->keys[idx+1] == 0) {
                                    prev_[level]->keys[idx+1] = prev_[level-1]->keys[0];
                                    prev_[level]->next[idx+1] = prev_[level-1];
                                    prev_[level]->N_key = ARR_SIZE / 2 + 1;
                                } else {
                                    Key update_key = prev_[level]->keys[0];
                                    prev_[level]->N_key = ARR_SIZE / 2;
                                    std::memmove(&prev_[level]->keys[idx+2], &prev_[level]->keys[idx+1], (prev_[level]->N_key - (idx+1)) * sizeof(Key));
                                    std::memmove(&prev_[level]->next[idx+2], &prev_[level]->next[idx+1], (prev_[level]->N_key - (idx+1)) * sizeof(prev_[level]->next[0]));
                                    shift_count++; // Signal.Jin
                                    prev_[level]->keys[idx+1] = prev_[level-1]->keys[0];
                                    prev_[level]->next[idx+1] = prev_[level-1];
                                    prev_[level]->N_key++;
                                    if (idx == -1) {
                                        for (int update = level+1; update < GetMaxHeight(); update++) {
                                            int idx = findMaxLessOrEqual(prev_[update]->forward->keys, update_key);
                                            if (prev_[update]->forward->keys[idx] == update_key) {
                                                prev_[update]->forward->keys[idx] = prev_[level]->forward->keys[0];
                                            }
                                            if (idx != 0) break;
                                        }
                                        stop_flag++;
                                    }
                                }
                                prev_[level] = add_node;
                                level++; // Keep tracking
                                if (cur_height < level) {
                                    max_height_++;
                                }
                            } else {
                                prev_[level]->N_key = ARR_SIZE / 2;
                                idx = idx - ARR_SIZE / 2;
                                Node* add_node = NewNode(prev_[level]->keys[ARR_SIZE/2]);
                                std::memcpy(add_node->keys, &prev_[level]->keys[ARR_SIZE/2], (ARR_SIZE/2) * sizeof(Key));
                                std::memcpy(add_node->next, &prev_[level]->next[ARR_SIZE/2], (ARR_SIZE/2) * sizeof(prev_[level]->next[0]));
                                std::memset(&prev_[level]->keys[ARR_SIZE/2], 0, (ARR_SIZE/2) * sizeof(Key));
                                shift_count++; // Signal.Jin
                                for (int i = ARR_SIZE/2; i < ARR_SIZE; i++) {
                                    prev_[level]->next[i] = nullptr;
                                }
                                add_node->N_key = ARR_SIZE/2 + 1;
                                add_node->forward = prev_[level]->forward;
                                prev_[level]->forward = add_node;
                                prev_[level] = add_node;
                                std::memmove(&prev_[level]->keys[idx+2], &prev_[level]->keys[idx+1], (prev_[level]->N_key - (idx+1)) * sizeof(Key));
                                std::memmove(&prev_[level]->next[idx+2], &prev_[level]->next[idx+1], (prev_[level]->N_key - (idx+1)) * sizeof(prev_[level]->next[0]));
                                shift_count++; // Signal.Jin
                                prev_[level]->keys[idx+1] = prev_[level-1]->keys[0];
                                prev_[level]->next[idx+1] = prev_[level-1];
                                level++;
                                if (cur_height < level) {
                                    max_height_++;
                                }
                            }
                        }
                    }   
                }
            }
        }

        if (stop_flag > 0) {
            break;
        }
    }
}

template<typename Key>
void SkipList<Key>::Insert_esplit(const Key& key) {
    Node* prev_[MAXHEIGHT];
    std::copy(std::begin(head_), std::end(head_), std::begin(prev_));
    int height = GetMaxHeight() - 1; // Using for search
    Node* x = head_[height]; // Use when searching

    if (x->forward != nullptr && compare_(x->forward->keys[0], key) <= 0) x = x->forward;

    while (true) { // Find the location which will insert the key using prev_ and head_
        prev_[height--] = x;
        if (height >= 0) {
            int n_key = x->N_key;
            if (n_key <= ARR_SIZE/2) {
                x = (x == head_[height + 1]) ? head_[height] : x->next[findMaxLessOrEqualLinearSIMD(x->keys, key, n_key)];
            } else {
                x = (x == head_[height + 1]) ? head_[height] : x->next[findMaxLessOrEqualBinary(x->keys, key)];
            }
        } else {
            break;
        }
    }

    // Do not allow duplicated key
    if (prev_[0] != head_[0]) {
        for (int i = 0; i < prev_[0]->N_key; i++) {
            if (compare_(prev_[0]->keys[i], key) == 0) {
                return;
            }
        }
    }
    
    // Do insert operation
    int level = 0;
    while (true) {
        int stop_flag = 0;
        int cur_height = GetMaxHeight() - 1;
        if (prev_[level] == head_[level] && prev_[level]->forward == nullptr) {
        // Case 1: There is no node in list and need to make a new node
            if (level == 0) { // Case 1-1: If insert into H0
                Node* Elist_node = NewNode(key);
                Elist_node->forward = prev_[level]->forward;
                prev_[level]->forward = Elist_node;
                break;
            } else { // Case 1-2: If not insert into H0
                if (cur_height < level) {
                    max_height_++;
                }
                if (prev_[level-1]->forward != nullptr && prev_[level-1] == head_[level-1]) {
                    Node* Elist_node = NewNode(prev_[level-1]->forward->keys[0]);
                    Elist_node->forward = prev_[level]->forward;
                    Elist_node->next[0] = prev_[level-1]->forward;
                    prev_[level]->forward = Elist_node;
                } else {
                    Node* Elist_node = NewNode(prev_[level-1]->keys[0]);
                    Elist_node->forward = prev_[level]->forward;
                    Elist_node->next[0] = prev_[level-1];
                    prev_[level]->forward = Elist_node;
                }
                break;
            }
        } else if (prev_[level] == head_[level]) {
            // Case 2: There is another node in list and need to make a new node
            if (prev_[level]->forward->N_key != ARR_SIZE) { // Case 2-1: Forward node has a room
                if (level == 0) { // Case 2-1-1: Insert into H0
                    int idx = findMaxLessOrEqual(prev_[level]->forward->keys, key);
                    if (idx == 0 && prev_[level]->forward->keys[0] > key) idx = -1;

                    if (prev_[level]->forward->keys[idx] == key) {
                        break;
                    } else if (prev_[level]->forward->keys[idx+1] == 0) {
                        prev_[level]->forward->keys[idx+1] = key;
                        prev_[level]->forward->N_key++;
                        if (prev_[level]->forward->N_key == ARR_SIZE) {
                            level++;
                            if (cur_height < level) {
                                max_height_++;
                            } 
                        } else break;
                    } else {
                        Key update_key = prev_[level]->forward->keys[0];
                        std::memmove(&prev_[level]->forward->keys[idx+2], &prev_[level]->forward->keys[idx+1], (prev_[level]->forward->N_key - (idx+1)) * sizeof(Key));
                        shift_count++; // Signal.Jin
                        prev_[level]->forward->keys[idx+1] = key;
                        prev_[level]->forward->N_key++;
                        if (idx == -1) {
                            for (int i = level+1; i < GetMaxHeight(); i++) {
                                if (prev_[i] != nullptr) {
                                    int idx = findMaxLessOrEqual(prev_[i]->forward->keys, update_key);
                                    if (prev_[i]->forward->keys[idx] == update_key) {
                                        prev_[i]->forward->keys[idx] = key;
                                    }
                                    if (idx != 0) break;
                                }
                            }
                        }

                        if (prev_[level]->forward->N_key == ARR_SIZE) {
                            level++;
                            if (cur_height < level) {
                                max_height_++;
                            } 
                        } else break;
                    }
                } else { // Case 2-1-2: Not insert into H0
                    int idx = findMaxLessOrEqual(prev_[level]->forward->keys, prev_[level-1]->forward->keys[0]);
                    if (idx == 0 && prev_[level]->forward->keys[0] > prev_[level-1]->forward->keys[0]) idx = -1;

                    if (prev_[level]->forward->keys[idx] == prev_[level-1]->forward->keys[0]) {
                        break;
                    } else if (prev_[level]->forward->keys[idx+1] == 0) {
                        prev_[level]->forward->keys[idx+1] = prev_[level-1]->forward->keys[0];
                        prev_[level]->forward->next[idx+1] = prev_[level-1]->forward;
                        prev_[level]->forward->N_key++;
                        if (prev_[level]->forward->N_key == ARR_SIZE) {
                            level++;
                            if (cur_height < level) {
                                max_height_++;
                            } 
                        } else break;
                    } else {
                        Key update_key = prev_[level]->forward->keys[0];
                        std::memmove(&prev_[level]->forward->keys[idx+2], &prev_[level]->forward->keys[idx+1], (prev_[level]->forward->N_key - (idx+1)) * sizeof(Key));
                        std::memmove(&prev_[level]->forward->next[idx+2], &prev_[level]->forward->next[idx+1], (prev_[level]->forward->N_key - (idx+1)) * sizeof(Key));
                        shift_count++; // Signal.Jin
                        prev_[level]->forward->keys[idx+1] = prev_[level-1]->forward->keys[0];
                        prev_[level]->forward->next[idx+1] = prev_[level-1]->forward;
                        prev_[level]->forward->N_key++;
                        if (idx == -1) {
                            for (int i = level+1; i < GetMaxHeight(); i++) {
                                if (prev_[i] != nullptr) {
                                    int idx = findMaxLessOrEqual(prev_[i]->forward->keys, update_key);
                                    if (prev_[i]->forward->keys[idx] == update_key) {
                                        prev_[i]->forward->keys[idx] = prev_[level-1]->forward->keys[0];
                                    }
                                    if (idx != 0) break;
                                }
                            }
                        }

                        if (prev_[level]->forward->N_key == ARR_SIZE) {
                            level++;
                            if (cur_height < level) {
                                max_height_++;
                            } 
                        } else break;
                    }
                }
            } else { // Case 2-2: Forward node has no room, so we need to make a new node
                // even-split operation
                if (level == 0) { // Case 2-2-1: Insert into H0
                    split_count++; // Signal.Jin
                    Node * add_node = NewNode(prev_[level]->forward->keys[ARR_SIZE/2]);
                    std::memcpy(add_node->keys, &prev_[level]->forward->keys[ARR_SIZE/2], (ARR_SIZE/2) * sizeof(Key));
                    std::memset(&prev_[level]->forward->keys[ARR_SIZE/2], 0, (ARR_SIZE/2) * sizeof(Key));
                    shift_count++; // Signal.Jin
                    add_node->N_key = ARR_SIZE/2;
                    add_node->forward = prev_[level]->forward->forward;
                    prev_[level]->forward->forward = add_node;
                    Key update_key = prev_[level]->forward->keys[0];
                    std::memmove(&prev_[level]->forward->keys[1], &prev_[level]->forward->keys[0], (ARR_SIZE/2) * sizeof(Key));
                    shift_count++;
                    prev_[level]->forward->N_key = (ARR_SIZE / 2) + 1;
                    prev_[level]->forward->keys[0] = key;
                    for (int update = level+1; update < GetMaxHeight(); update++) {
                        if (prev_[update]->forward != nullptr) {
                            int idx = findMaxLessOrEqual(prev_[update]->forward->keys, update_key);
                            if (prev_[update]->forward->keys[idx] == update_key) {
                                prev_[update]->forward->keys[idx] = key;
                            }
                            if (idx != 0) break;
                        }
                    }
                    
                    prev_[level] = add_node;
                    for (int i = level + 1; i < GetMaxHeight(); i++) {
                        prev_[i] = prev_[i]->forward;
                    }
                    level++; // Keep tracking
                    if (cur_height < level) {
                        max_height_++;
                    }
                } else { // Case 2-2-2: Not insert into H0
                    split_count++; // Signal.Jin
                    Node * add_node = NewNode(prev_[level]->forward->keys[ARR_SIZE/2]);
                    std::memcpy(add_node->keys, &prev_[level]->forward->keys[ARR_SIZE/2], (ARR_SIZE/2) * sizeof(Key));
                    std::memcpy(add_node->next, &prev_[level]->forward->next[ARR_SIZE/2], (ARR_SIZE/2) * sizeof(prev_[level]->forward->next[0]));
                    std::memset(&prev_[level]->forward->keys[ARR_SIZE/2], 0, (ARR_SIZE/2) * sizeof(Key));
                    shift_count++; // Signal.Jin
                    add_node->N_key = ARR_SIZE/2;
                    add_node->forward = prev_[level]->forward->forward;
                    prev_[level]->forward->forward = add_node;

                    std::memmove(&prev_[level]->forward->keys[1], &prev_[level]->forward->keys[0], (ARR_SIZE/2) * sizeof(Key));
                    shift_count++;
                    Key update_key = prev_[level]->forward->keys[0];
                    prev_[level]->forward->N_key = (ARR_SIZE / 2) + 1;
                    prev_[level]->forward->keys[0] = prev_[level-1]->forward->keys[0];
                    for (int update = level+1; update < GetMaxHeight(); update++) {
                        int idx = findMaxLessOrEqual(prev_[update]->forward->keys, update_key);
                        if (prev_[update]->forward->keys[idx] == update_key) {
                            prev_[update]->forward->keys[idx] = prev_[level-1]->forward->keys[0];
                        }
                        if (idx != 0) break;
                    }

                    prev_[level] = add_node;
                    for (int i = level + 1; i < GetMaxHeight(); i++) {
                        prev_[i] = prev_[i]->forward;
                    }
                    level++; // Keep tracking
                    if (cur_height < level) {
                        max_height_++;
                    }
                }
            }
        } else {
            // Case 3: New node must be inserted between nodes or into prev_ node (not head_)
            if (prev_[level]->N_key != ARR_SIZE) { // Case 3-1: prev_ node has a room, so we insert into that node
                if (level == 0) { // Case 3-1-1: Insert into H0
                    int idx = findMaxLessOrEqual(prev_[level]->keys, key);
                    if (idx == 0 && prev_[level]->keys[0] > key) idx = -1;
                    if (prev_[level]->keys[idx] == key) {
                        stop_flag++;
                    } else {
                        if (prev_[level]->keys[idx+1] == 0) {
                            prev_[level]->keys[idx+1] = key;
                            prev_[level]->N_key++;    
                        } else {
                            std::memmove(&prev_[level]->keys[idx+2], &prev_[level]->keys[idx+1], (prev_[level]->N_key - (idx+1)) * sizeof(Key));
                            shift_count++; // Signal.Jin
                            prev_[level]->keys[idx+1] = key;
                            prev_[level]->N_key++;
                        }
                        if (prev_[level]->N_key == ARR_SIZE && prev_[level+1] == head_[level+1]) {
                            level++;
                            if (cur_height < level) {
                                max_height_++;
                            }
                        } else stop_flag++;
                    }
                } else { // Case 3-1-2: Not insert into H0
                    int idx = findMaxLessOrEqual(prev_[level]->keys, prev_[level-1]->keys[0]);
                    if (idx == 0 && prev_[level]->keys[0] > prev_[level-1]->keys[0]) idx = -1;
                    if (prev_[level]->keys[idx] == prev_[level-1]->keys[0]) {
                        stop_flag++;
                    } else {
                        if (prev_[level]->keys[idx+1] == 0) {
                            prev_[level]->keys[idx+1] = prev_[level-1]->keys[0];
                            prev_[level]->next[idx+1] = prev_[level-1];
                            prev_[level]->N_key++;
                        } else {
                            std::memmove(&prev_[level]->keys[idx+2], &prev_[level]->keys[idx+1], (prev_[level]->N_key - (idx+1)) * sizeof(Key));
                            std::memmove(&prev_[level]->next[idx+2], &prev_[level]->next[idx+1], (prev_[level]->N_key - (idx+1)) * sizeof(Key));
                            shift_count++; // Signal.Jin
                            prev_[level]->keys[idx+1] = prev_[level-1]->keys[0];
                            prev_[level]->next[idx+1] = prev_[level-1];
                            prev_[level]->N_key++;
                        }
                    }
                    if (prev_[level]->N_key == ARR_SIZE && prev_[level+1] == head_[level+1]) {
                        level++;
                        if (cur_height < level) {
                            max_height_++;
                        }
                    } else stop_flag++;
                }
            } else { // Case 3-2: prev_ node has no room, so we need to make a new node
                // When creating a new node, change prev to the newly created node.
                // even-split operation
                if (level == 0) { // Case 3-2-1: Insert into H0
                    split_count++; // Signal.Jin
                    int idx = findMaxLessOrEqual(prev_[level]->keys, key);
                    {
                        if (idx < ARR_SIZE / 2) {
                            Node* add_node = NewNode(prev_[level]->keys[ARR_SIZE/2]);
                            std::memcpy(add_node->keys, &prev_[level]->keys[ARR_SIZE/2], (ARR_SIZE/2) * sizeof(Key));
                            std::memset(&prev_[level]->keys[ARR_SIZE/2], 0, (ARR_SIZE/2) * sizeof(Key));
                            shift_count++; // Signal.Jin
                            add_node->N_key = ARR_SIZE/2;
                            add_node->forward = prev_[level]->forward;
                            prev_[level]->forward = add_node;
                            if (idx == ARR_SIZE / 2 - 1) {
                                prev_[level]->keys[idx + 1] = key;
                                prev_[level]->N_key = ARR_SIZE / 2 + 1;
                            } else {
                                Key update_key = prev_[level]->keys[0];
                                prev_[level]->N_key = ARR_SIZE / 2;
                                std::memmove(&prev_[level]->keys[idx+2], &prev_[level]->keys[idx+1], (prev_[level]->N_key - (idx+1)) * sizeof(Key));
                                shift_count++; // Signal.Jin
                                prev_[level]->keys[idx+1] = key;
                                prev_[level]->N_key++;
                                if (idx == -1) {
                                    for (int update = level+1; update < GetMaxHeight(); update++) {
                                        int idx = findMaxLessOrEqual(prev_[update]->forward->keys, update_key);
                                        //printf("Case 3-2-1 (2) Index = %d\n", idx);
                                        if (prev_[update]->forward->keys[idx] == update_key) {
                                            prev_[update]->forward->keys[idx] = key;
                                        }
                                        if (idx != 0) break;
                                    }
                                    stop_flag++;
                                }
                            }
                            prev_[level] = add_node;
                            level++; // Keep tracking
                            if (cur_height < level) {
                                max_height_++;
                            }
                        } else {
                            prev_[level]->N_key = ARR_SIZE / 2;
                            idx = idx - ARR_SIZE / 2;
                            Node* add_node = NewNode(prev_[level]->keys[ARR_SIZE/2]);
                            std::memcpy(add_node->keys, &prev_[level]->keys[ARR_SIZE/2], (ARR_SIZE/2) * sizeof(Key));
                            std::memset(&prev_[level]->keys[ARR_SIZE/2], 0, (ARR_SIZE/2) * sizeof(Key));
                            shift_count++; // Signal.Jin
                            add_node->N_key = ARR_SIZE/2 + 1;
                            add_node->forward = prev_[level]->forward;
                            prev_[level]->forward = add_node;
                            prev_[level] = add_node;
                            std::memmove(&prev_[level]->keys[idx+2], &prev_[level]->keys[idx+1], (prev_[level]->N_key - (idx+1)) * sizeof(Key));
                            shift_count++; // Signal.Jin
                            prev_[level]->keys[idx+1] = key;
                            level++;
                            if (cur_height < level) {
                                max_height_++;
                            }
                        }
                    }
                } else { // Case 3-2-2: Not insert into H0
                    split_count++; // Signal.Jin
                    int idx = findMaxLessOrEqual(prev_[level]->keys, prev_[level-1]->keys[0]);
                    {
                        if (idx < ARR_SIZE / 2) {
                            Node* add_node = NewNode(prev_[level]->keys[ARR_SIZE/2]);
                            std::memcpy(add_node->keys, &prev_[level]->keys[ARR_SIZE/2], (ARR_SIZE/2) * sizeof(Key));
                            std::memcpy(add_node->next, &prev_[level]->next[ARR_SIZE/2], (ARR_SIZE/2) * sizeof(prev_[level]->next[0]));
                            std::memset(&prev_[level]->keys[ARR_SIZE/2], 0, (ARR_SIZE/2) * sizeof(Key));
                            shift_count++; // Signal.Jin
                            for (int i = ARR_SIZE/2; i < ARR_SIZE; i++) {
                                prev_[level]->next[i] = nullptr;
                            }
                            add_node->N_key = ARR_SIZE/2;
                            add_node->forward = prev_[level]->forward;
                            prev_[level]->forward = add_node;
                            if (idx == ARR_SIZE / 2 - 1) {
                                prev_[level]->keys[idx+1] = prev_[level-1]->keys[0];
                                prev_[level]->next[idx+1] = prev_[level-1];
                                prev_[level]->N_key = ARR_SIZE / 2 + 1;
                            } else {
                                Key update_key = prev_[level]->keys[0];
                                prev_[level]->N_key = ARR_SIZE / 2;
                                std::memmove(&prev_[level]->keys[idx+2], &prev_[level]->keys[idx+1], (prev_[level]->N_key - (idx+1)) * sizeof(Key));
                                std::memmove(&prev_[level]->next[idx+2], &prev_[level]->next[idx+1], (prev_[level]->N_key - (idx+1)) * sizeof(prev_[level]->next[0]));
                                shift_count++; // Signal.Jin
                                prev_[level]->keys[idx+1] = prev_[level-1]->keys[0];
                                prev_[level]->next[idx+1] = prev_[level-1];
                                prev_[level]->N_key++;
                                if (idx == -1) {
                                    for (int update = level+1; update < GetMaxHeight(); update++) {
                                        int idx = findMaxLessOrEqual(prev_[update]->forward->keys, update_key);
                                        if (prev_[update]->forward->keys[idx] == update_key) {
                                            prev_[update]->forward->keys[idx] = prev_[level]->forward->keys[0];
                                        }
                                        if (idx != 0) break;
                                    }
                                    stop_flag++;
                                }
                            }
                            prev_[level] = add_node;
                            level++; // Keep tracking
                            if (cur_height < level) {
                                max_height_++;
                            }
                        } else {
                            prev_[level]->N_key = ARR_SIZE / 2;
                            idx = idx - ARR_SIZE / 2;
                            Node* add_node = NewNode(prev_[level]->keys[ARR_SIZE/2]);
                            std::memcpy(add_node->keys, &prev_[level]->keys[ARR_SIZE/2], (ARR_SIZE/2) * sizeof(Key));
                            std::memcpy(add_node->next, &prev_[level]->next[ARR_SIZE/2], (ARR_SIZE/2) * sizeof(prev_[level]->next[0]));
                            std::memset(&prev_[level]->keys[ARR_SIZE/2], 0, (ARR_SIZE/2) * sizeof(Key));
                            shift_count++; // Signal.Jin
                            for (int i = ARR_SIZE/2; i < ARR_SIZE; i++) {
                                prev_[level]->next[i] = nullptr;
                            }
                            add_node->N_key = ARR_SIZE/2 + 1;
                            add_node->forward = prev_[level]->forward;
                            prev_[level]->forward = add_node;
                            prev_[level] = add_node;
                            std::memmove(&prev_[level]->keys[idx+2], &prev_[level]->keys[idx+1], (prev_[level]->N_key - (idx+1)) * sizeof(Key));
                            std::memmove(&prev_[level]->next[idx+2], &prev_[level]->next[idx+1], (prev_[level]->N_key - (idx+1)) * sizeof(prev_[level]->next[0]));
                            shift_count++; // Signal.Jin
                            prev_[level]->keys[idx+1] = prev_[level-1]->keys[0];
                            prev_[level]->next[idx+1] = prev_[level-1];
                            level++;
                            if (cur_height < level) {
                                max_height_++;
                            }
                        }
                    }
                }
            }
        }

        if (stop_flag > 0) {
            break;
        }
    }
}

template<typename Key>
void SkipList<Key>::Insert_Search(const Key& key) {
    Node* prev_[MAXHEIGHT];
    std::copy(std::begin(head_), std::end(head_), std::begin(prev_));
    int height = GetMaxHeight() - 1; // Using for search
    Node* x = head_[height]; // Use when searching

    if (x->forward != nullptr && compare_(x->forward->keys[0], key) <= 0) x = x->forward;

    while (true) { // Find the location which will insert the key using prev_ and head_
        prev_[height--] = x;
        if (height >= 0) {
            int n_key = x->N_key;
            if (n_key <= ARR_SIZE/2) {
                x = (x == head_[height + 1]) ? head_[height] : x->next[findMaxLessOrEqualLinearSIMD(x->keys, key, n_key)];
            } else {
                x = (x == head_[height + 1]) ? head_[height] : x->next[findMaxLessOrEqualBinary(x->keys, key)];
            }
        } else {
            break;
        }
    }
    
    // Do not allow duplicated key
    if (prev_[0] != head_[0]) {
        for (int i = 0; i < prev_[0]->N_key; i++) {
            if (compare_(prev_[0]->keys[i], key) == 0) {
                return;
            }
        }
    }
    
    // Do insert operation
    int level = 0;
    while (true) {
        int stop_flag = 0;
        int cur_height = GetMaxHeight() - 1;
        if (prev_[level] == head_[level] && prev_[level]->forward == nullptr) {
        // Case 1: There is no node in list and need to make a new node
            if (level == 0) { // Case 1-1: If insert into H0
                Node* Elist_node = NewNode(key);
                Elist_node->forward = prev_[level]->forward;
                prev_[level]->forward = Elist_node;
                break;
            } else { // Case 1-2: If not insert into H0
                if (cur_height < level) {
                    max_height_++;
                }
                if (prev_[level-1]->forward != nullptr && prev_[level-1] == head_[level-1]) {
                    Node* Elist_node = NewNode(prev_[level-1]->forward->keys[0]);
                    Elist_node->forward = prev_[level]->forward;
                    Elist_node->next[0] = prev_[level-1]->forward;
                    prev_[level]->forward = Elist_node;
                } else {
                    Node* Elist_node = NewNode(prev_[level-1]->keys[0]);
                    Elist_node->forward = prev_[level]->forward;
                    Elist_node->next[0] = prev_[level-1];
                    prev_[level]->forward = Elist_node;
                }
                break;
            }
        } else if (prev_[level] == head_[level]) {
            // Case 2: There is another node in list and need to make a new node
            if (prev_[level]->forward->N_key != ARR_SIZE) { // Case 2-1: Forward node has a room
                if (level == 0) { // Case 2-1-1: Insert into H0
                    int idx = findMaxLessOrEqual(prev_[level]->forward->keys, key);
                    if (idx == 0 && prev_[level]->forward->keys[0] > key) idx = -1;

                    if (prev_[level]->forward->keys[idx] == key) {
                        break;
                    } else if (prev_[level]->forward->keys[idx+1] == 0) {
                        prev_[level]->forward->keys[idx+1] = key;
                        prev_[level]->forward->N_key++;
                        if (prev_[level]->forward->N_key == ARR_SIZE) {
                            level++;
                            if (cur_height < level) {
                                max_height_++;
                            } 
                        } else break;
                    } else {
                        Key update_key = prev_[level]->forward->keys[0];
                        std::memmove(&prev_[level]->forward->keys[idx+2], &prev_[level]->forward->keys[idx+1], (prev_[level]->forward->N_key - (idx+1)) * sizeof(Key));
                        shift_count++; // Signal.Jin
                        prev_[level]->forward->keys[idx+1] = key;
                        prev_[level]->forward->N_key++;
                        if (idx == -1) {
                            for (int i = level+1; i < GetMaxHeight(); i++) {
                                if (prev_[i] != nullptr) {
                                    int idx = findMaxLessOrEqual(prev_[i]->forward->keys, update_key);
                                    if (prev_[i]->forward->keys[idx] == update_key) {
                                        prev_[i]->forward->keys[idx] = key;
                                    }
                                    if (idx != 0) break;
                                }
                            }
                        }

                        if (prev_[level]->forward->N_key == ARR_SIZE) {
                            level++;
                            if (cur_height < level) {
                                max_height_++;
                            } 
                        } else break;
                    }
                } else { // Case 2-1-2: Not insert into H0
                    int idx = findMaxLessOrEqual(prev_[level]->forward->keys, prev_[level-1]->forward->keys[0]);
                    if (idx == 0 && prev_[level]->forward->keys[0] > prev_[level-1]->forward->keys[0]) idx = -1;

                    if (prev_[level]->forward->keys[idx] == prev_[level-1]->forward->keys[0]) {
                        break;
                    } else if (prev_[level]->forward->keys[idx+1] == 0) {
                        prev_[level]->forward->keys[idx+1] = prev_[level-1]->forward->keys[0];
                        prev_[level]->forward->next[idx+1] = prev_[level-1]->forward;
                        prev_[level]->forward->N_key++;
                        if (prev_[level]->forward->N_key == ARR_SIZE) {
                            level++;
                            if (cur_height < level) {
                                max_height_++;
                            } 
                        } else break;
                    } else {
                        Key update_key = prev_[level]->forward->keys[0];
                        std::memmove(&prev_[level]->forward->keys[idx+2], &prev_[level]->forward->keys[idx+1], (prev_[level]->forward->N_key - (idx+1)) * sizeof(Key));
                        std::memmove(&prev_[level]->forward->next[idx+2], &prev_[level]->forward->next[idx+1], (prev_[level]->forward->N_key - (idx+1)) * sizeof(Key));
                        shift_count++; // Signal.Jin
                        prev_[level]->forward->keys[idx+1] = prev_[level-1]->forward->keys[0];
                        prev_[level]->forward->next[idx+1] = prev_[level-1]->forward;
                        prev_[level]->forward->N_key++;
                        if (idx == -1) {
                            for (int i = level+1; i < GetMaxHeight(); i++) {
                                if (prev_[i] != nullptr) {
                                    int idx = findMaxLessOrEqual(prev_[i]->forward->keys, update_key);
                                    if (prev_[i]->forward->keys[idx] == update_key) {
                                        prev_[i]->forward->keys[idx] = prev_[level-1]->forward->keys[0];
                                    }
                                    if (idx != 0) break;
                                }
                            }
                        }

                        if (prev_[level]->forward->N_key == ARR_SIZE) {
                            level++;
                            if (cur_height < level) {
                                max_height_++;
                            } 
                        } else break;
                    }
                }
            } else { // Case 2-2: Forward node has no room, so we need to make a new node
                if (level == 0) { // Case 2-2-1: Insert into H0
                    Node* add_node = NewNode(key);
                    add_node->forward = prev_[level]->forward;
                    prev_[level]->forward = add_node;
                    break;
                } else { // Case 2-2-2: Not insert into H0
                    if (prev_[level-1] == head_[level-1]) {
                        Node* add_node = NewNode(prev_[level-1]->forward->keys[0]);
                        add_node->forward = prev_[level]->forward;
                        add_node->next[0] = prev_[level-1]->forward;
                        prev_[level]->forward = add_node;
                    } else {
                        Node* add_node = NewNode(prev_[level-1]->keys[0]);
                        add_node->forward = prev_[level]->forward;
                        add_node->next[0] = prev_[level-1];
                        prev_[level]->forward = add_node;
                    }
                    break;
                }
            }
        } else {
            // Case 3: New node must be inserted between nodes or into prev_ node (not head_)
            if (prev_[level]->N_key != ARR_SIZE) { // Case 3-1: prev_ node has a room, so we insert into that node
                if (level == 0) { // Case 3-1-1: Insert into H0
                    for (int shift = 0; shift < prev_[level]->N_key; shift++) {
                        if (prev_[level]->keys[shift] == key) {
                            stop_flag++;
                            break;
                        } else if (prev_[level]->keys[shift] <= key && key < prev_[level]->keys[shift+1]) {
                            for (int do_shift = prev_[level]->N_key-1; do_shift > shift; do_shift--) {
                                prev_[level]->keys[do_shift+1] = prev_[level]->keys[do_shift];
                            }
                            shift_count++; // Signal.Jin
                            prev_[level]->keys[shift+1] = key;
                            prev_[level]->N_key++;
                            if (prev_[level]->N_key == ARR_SIZE && prev_[level+1] == head_[level+1]) {
                                level++;
                                if (cur_height < level) {
                                    max_height_++;
                                }
                            } else stop_flag++;
                            break;
                        } else if (prev_[level]->keys[shift+1] == 0) {
                            prev_[level]->keys[shift+1] = key;
                            prev_[level]->N_key++;
                            if (prev_[level]->N_key == ARR_SIZE && prev_[level+1] == head_[level+1]) {
                                level++;
                                if (cur_height < level) {
                                    max_height_++;
                                }
                            } else stop_flag++;
                            break;
                        }
                    }
                } else { // Case 3-1-2: Not insert into H0
                    for (int shift = 0; shift < prev_[level]->N_key; shift++) {
                        if (prev_[level]->keys[shift] == prev_[level-1]->keys[0]) {
                            stop_flag++;
                            break;
                        } else if (prev_[level]->keys[shift] < prev_[level-1]->keys[0] && prev_[level-1]->keys[0] < prev_[level]->keys[shift+1]) {
                            for (int do_shift = prev_[level]->N_key-1; do_shift > shift; do_shift--) {
                                prev_[level]->keys[do_shift+1] = prev_[level]->keys[do_shift];
                                prev_[level]->next[do_shift+1] = prev_[level]->next[do_shift];
                            }
                            shift_count++; // Signal.Jin
                            prev_[level]->keys[shift+1] = prev_[level-1]->keys[0];
                            prev_[level]->next[shift+1] = prev_[level-1];
                            prev_[level]->N_key++;
                            if (prev_[level]->N_key == ARR_SIZE && prev_[level+1] == head_[level+1]) {
                                level++;
                                if (cur_height < level) {
                                    max_height_++;
                                }
                            } else stop_flag++;
                            break;
                        } else if (prev_[level]->keys[shift+1] == 0) {
                            prev_[level]->keys[shift+1] = prev_[level-1]->keys[0];
                            prev_[level]->next[shift+1] = prev_[level-1];
                            prev_[level]->N_key++;
                            if (prev_[level]->N_key == ARR_SIZE && prev_[level+1] == head_[level+1]) {
                                level++;
                                if (cur_height < level) {
                                    max_height_++;
                                }
                            } else stop_flag++;
                            break;
                        }
                    }
                }
            } else { // Case 3-2: prev_ node has no room, so we need to make a new node
                // When creating a new node, change prev to the newly created node.
                if (level == 0) { // Case 3-2-1: Insert into H0
                    int no_room = 0;
                    if (prev_[level]->keys[ARR_SIZE-1] < key && prev_[level]->keys[ARR_SIZE-1] != 0) {
                        no_room++;
                    } else {
                        for (int shift = 0; shift < prev_[level]->N_key-1; shift++) {
                            if (prev_[level]->keys[shift] <= key && key < prev_[level]->keys[shift+1]) {
                                Key temp_key = prev_[level]->keys[ARR_SIZE-1];
                                for (int do_shift = prev_[level]->N_key-2; do_shift > shift; do_shift--) {
                                    prev_[level]->keys[do_shift+1] = prev_[level]->keys[do_shift];
                                }
                                shift_count++; // Signal.Jin
                                prev_[level]->keys[shift+1] = key;
                                if (prev_[level]->forward != nullptr && prev_[level]->forward->N_key < ARR_SIZE) {
                                    Key update_key = prev_[level]->forward->keys[0];
                                    for (int d_shift = prev_[level]->forward->N_key; d_shift > 0; d_shift--) {
                                        prev_[level]->forward->keys[d_shift] = prev_[level]->forward->keys[d_shift-1];
                                    }
                                    shift_count++; // Signal.Jin
                                    prev_[level]->forward->keys[0] = temp_key;
                                    prev_[level]->forward->N_key++;
                                    if (false) {
                                        level++;
                                        if (cur_height < level) {
                                            max_height_++;
                                        } 
                                    } else {
                                        for (int update = level+1; update < GetMaxHeight(); update++) {
                                            int loop = 0;
                                            for (loop = 0; loop < prev_[update]->N_key; loop++) {
                                                if (prev_[update]->keys[loop] == update_key) {
                                                    prev_[update]->keys[loop] = temp_key;
                                                    break;
                                                }
                                            }
                                            if (prev_[update]->forward != nullptr) {
                                                for (loop = 0; loop < prev_[update]->forward->N_key; loop++) {
                                                    if (prev_[update]->forward->keys[loop] == update_key) {
                                                        prev_[update]->forward->keys[loop] = temp_key;
                                                        break;
                                                    }
                                                }
                                            }
                                            if (loop != 0) break;
                                        }
                                        stop_flag++;
                                    }
                                    break;
                                } else {
                                    Node* add_node = NewNode(temp_key);
                                    add_node->forward = prev_[level]->forward;
                                    prev_[level]->forward = add_node;
                                    prev_[level] = add_node;
                                    level++;
                                    if (cur_height < level) {
                                        max_height_++;
                                    }
                                    break;
                                }
                            }
                        }
                    }
                    if (no_room != 0) {
                        if (prev_[level]->forward != nullptr && prev_[level]->forward->N_key < ARR_SIZE) {
                            Key update_key = prev_[level]->forward->keys[0];
                            for (int d_shift = prev_[level]->forward->N_key; d_shift > 0; d_shift--) {
                                prev_[level]->forward->keys[d_shift] = prev_[level]->forward->keys[d_shift-1];
                            }
                            shift_count++; // Signal.Jin
                            prev_[level]->forward->keys[0] = key;
                            prev_[level]->forward->N_key++;
                            if (false) {
                                level++;
                                if (cur_height < level) {
                                    max_height_++;
                                } 
                            } else {
                                for (int update = level+1; update < GetMaxHeight(); update++) {
                                    int loop = 0;
                                    for (loop = 0; loop < prev_[update]->N_key; loop++) {
                                        if (prev_[update]->keys[loop] == update_key) {
                                            prev_[update]->keys[loop] = key;
                                            break;
                                        }
                                    }
                                    if (prev_[update]->forward != nullptr) {
                                        for (loop = 0; loop < prev_[update]->forward->N_key; loop++) {
                                            if (prev_[update]->forward->keys[loop] == update_key) {
                                                prev_[update]->forward->keys[loop] = key;
                                                break;
                                            }
                                        }
                                    }
                                    if (loop != 0) break;
                                }
                                stop_flag++;
                            }
                            break;
                        } else {
                            Node* add_node = NewNode(key);
                            add_node->forward = prev_[level]->forward;
                            prev_[level]->forward = add_node;
                            prev_[level] = add_node;
                            level++;
                            if (cur_height < level) {
                                max_height_++;
                            }
                        }
                    }
                } else { // Case 3-2-2: Not insert into H0
                    int no_room = 0;
                    if (prev_[level]->keys[ARR_SIZE-1] < prev_[level-1]->keys[0] && prev_[level]->keys[ARR_SIZE-1] != 0) {
                        no_room++;
                    } else {
                        for (int shift = 0; shift < prev_[level]->N_key-1; shift++) {
                            if (prev_[level]->keys[shift] <= prev_[level-1]->keys[0] && prev_[level-1]->keys[0] < prev_[level]->keys[shift+1]) {
                                Key temp_key = prev_[level]->keys[ARR_SIZE-1];
                                Node* temp_next = prev_[level]->next[ARR_SIZE-1];
                                for (int do_shift = prev_[level]->N_key-2; do_shift > shift; do_shift--) {
                                    prev_[level]->keys[do_shift+1] = prev_[level]->keys[do_shift];
                                    prev_[level]->next[do_shift+1] = prev_[level]->next[do_shift];
                                }
                                prev_[level]->keys[shift+1] = prev_[level-1]->keys[0];
                                prev_[level]->next[shift+1] = prev_[level-1];
                                if (prev_[level]->forward != nullptr && prev_[level]->forward->N_key < ARR_SIZE) {
                                    Key update_key = prev_[level]->forward->keys[0];
                                    for (int d_shift = prev_[level]->forward->N_key; d_shift > 0; d_shift--) {
                                        prev_[level]->forward->keys[d_shift] = prev_[level]->forward->keys[d_shift-1];
                                        prev_[level]->forward->next[d_shift] = prev_[level]->forward->next[d_shift-1];
                                    }
                                    shift_count++; // Signal.Jin
                                    prev_[level]->forward->keys[0] = temp_key;
                                    prev_[level]->forward->next[0] = temp_next;
                                    prev_[level]->forward->N_key++;
                                    if (false) {
                                        level++;
                                        if (cur_height < level) {
                                            max_height_++;
                                        } 
                                    } else {
                                        for (int update = level+1; update < GetMaxHeight(); update++) {
                                            int loop = 0;
                                            for (loop = 0; loop < prev_[update]->N_key; loop++) {
                                                if (prev_[update]->keys[loop] == update_key) {
                                                    prev_[update]->keys[loop] = temp_key;
                                                    break;
                                                }
                                            }
                                            if (prev_[update]->forward != nullptr) {
                                                for (loop = 0; loop < prev_[update]->forward->N_key; loop++) {
                                                    if (prev_[update]->forward->keys[loop] == update_key) {
                                                        prev_[update]->forward->keys[loop] = temp_key;
                                                        break;
                                                    }
                                                }
                                            }
                                            if (loop != 0) break;
                                        }
                                        stop_flag++;
                                    }
                                    break;
                                } else {
                                    Node* add_node = NewNode(temp_key);
                                    add_node->forward = prev_[level]->forward;
                                    add_node->next[0] = temp_next;
                                    prev_[level]->forward = add_node;
                                    prev_[level] = add_node;
                                    level++;
                                    if (cur_height < level) {
                                        max_height_++;
                                    }
                                    break;
                                }
                            }
                        }
                    }
                    if (no_room != 0) {
                        if (prev_[level]->forward != nullptr && prev_[level]->forward->N_key < ARR_SIZE) {
                            Key update_key = prev_[level]->forward->keys[0];
                            for (int d_shift = prev_[level]->forward->N_key; d_shift > 0; d_shift--) {
                                prev_[level]->forward->keys[d_shift] = prev_[level]->forward->keys[d_shift-1];
                                prev_[level]->forward->next[d_shift] = prev_[level]->forward->next[d_shift-1];
                            }
                            shift_count++; // Signal.Jin
                            prev_[level]->forward->keys[0] = prev_[level-1]->keys[0];
                            prev_[level]->forward->next[0] = prev_[level-1];
                            prev_[level]->forward->N_key++;
                            if (false) {
                                level++;
                                if (cur_height < level) {
                                    max_height_++;
                                } 
                            } else {
                                for (int update = level+1; update < GetMaxHeight(); update++) {
                                    int loop = 0;
                                    for (loop = 0; loop < prev_[update]->N_key; loop++) {
                                        if (prev_[update]->keys[loop] == update_key) {
                                            prev_[update]->keys[loop] = prev_[level]->forward->keys[0];
                                            break;
                                        }
                                    }
                                    if (prev_[update]->forward != nullptr) {
                                        for (loop = 0; loop < prev_[update]->forward->N_key; loop++) {
                                            if (prev_[update]->forward->keys[loop] == update_key) {
                                                prev_[update]->forward->keys[loop] = prev_[level]->forward->keys[0];
                                                break;
                                            }
                                        }
                                    }
                                    if (loop != 0) break;
                                }
                                stop_flag++;
                            }
                            break;
                        } else {
                            Node* add_node = NewNode(prev_[level-1]->keys[0]);
                            add_node->forward = prev_[level]->forward;
                            add_node->next[0] = prev_[level-1];
                            prev_[level]->forward = add_node;
                            prev_[level] = add_node;
                            level++;
                            if (cur_height < level) {
                                max_height_++;
                            }
                        }
                    }    
                }
            }
        }

        if (stop_flag > 0) {
            break;
        }
    }
}

template<typename Key>
void SkipList<Key>::Insert_Raise(const Key& key) {
    Node* prev_[MAXHEIGHT];
    std::copy(std::begin(head_), std::end(head_), std::begin(prev_));
    int height = GetMaxHeight() - 1; // Using for search
    Node* x = head_[height]; // Use when searching
    
    if (x->forward != nullptr && compare_(x->forward->keys[0], key) <= 0) x = x->forward;

    while (true) { // Find the location which will insert the key using prev_ and head_
        prev_[height--] = x;
        if (height >= 0) {
            int n_key = x->N_key;
            x = (x == head_[height + 1]) ? head_[height] : x->next[findMaxLessOrEqualLinear(x->keys, key, n_key)];
        } else {
            break;
        }
    }
    
    // Do not allow duplicated key
    if (prev_[0] != head_[0]) {
        for (int i = 0; i < prev_[0]->N_key; i++) {
            if (compare_(prev_[0]->keys[i], key) == 0) {
                return;
            }
        }
    }
    
    // Do insert operation
    int level = 0;
    while (true) {
        int stop_flag = 0;
        int cur_height = GetMaxHeight() - 1;
        if (prev_[level] == head_[level] && prev_[level]->forward == nullptr) {
        // Case 1: There is no node in list and need to make a new node
            if (level == 0) { // Case 1-1: If insert into H0
                Node* Elist_node = NewNode(key);
                Elist_node->forward = prev_[level]->forward;
                prev_[level]->forward = Elist_node;
                break;
            } else { // Case 1-2: If not insert into H0
                if (cur_height < level) {
                    max_height_++;
                }
                if (prev_[level-1]->forward != nullptr && prev_[level-1] == head_[level-1]) {
                    Node* Elist_node = NewNode(prev_[level-1]->forward->keys[0]);
                    Elist_node->forward = prev_[level]->forward;
                    Elist_node->next[0] = prev_[level-1]->forward;
                    prev_[level]->forward = Elist_node;
                } else {
                    Node* Elist_node = NewNode(prev_[level-1]->keys[0]);
                    Elist_node->forward = prev_[level]->forward;
                    Elist_node->next[0] = prev_[level-1];
                    prev_[level]->forward = Elist_node;
                }
                break;
            }
        } else if (prev_[level] == head_[level]) {
            // Case 2: There is another node in list and need to make a new node
            if (prev_[level]->forward->N_key != ARR_SIZE) { // Case 2-1: Forward node has a room
                if (level == 0) { // Case 2-1-1: Insert into H0
                    int idx = findMaxLessOrEqualLinear(prev_[level]->forward->keys, key, prev_[level]->forward->N_key);
                    if (idx == 0 && prev_[level]->forward->keys[0] > key) idx = -1;

                    if (prev_[level]->forward->keys[idx] == key) {
                        break;
                    } else if (prev_[level]->forward->keys[idx+1] == 0) {
                        prev_[level]->forward->keys[idx+1] = key;
                        prev_[level]->forward->N_key++;
                        if (prev_[level]->forward->N_key == ARR_SIZE) {
                            level++;
                            if (cur_height < level) {
                                max_height_++;
                            } 
                        } else break;
                    } else {
                        Key update_key = prev_[level]->forward->keys[0];
                        std::memmove(&prev_[level]->forward->keys[idx+2], &prev_[level]->forward->keys[idx+1], (prev_[level]->forward->N_key - (idx+1)) * sizeof(Key));
                        shift_count++; // Signal.Jin
                        prev_[level]->forward->keys[idx+1] = key;
                        prev_[level]->forward->N_key++;
                        if (idx == -1) {
                            for (int i = level+1; i < GetMaxHeight(); i++) {
                                if (prev_[i] != nullptr) {
                                    int idx = findMaxLessOrEqualLinear(prev_[i]->forward->keys, update_key, prev_[i]->forward->N_key);
                                    if (prev_[i]->forward->keys[idx] == update_key) {
                                        prev_[i]->forward->keys[idx] = key;
                                    }
                                    if (idx != 0) break;
                                }
                            }
                        }

                        if (prev_[level]->forward->N_key == ARR_SIZE) {
                            level++;
                            if (cur_height < level) {
                                max_height_++;
                            } 
                        } else break;
                    }
                } else { // Case 2-1-2: Not insert into H0
                    int idx = findMaxLessOrEqualLinear(prev_[level]->forward->keys, prev_[level-1]->forward->keys[0], prev_[level]->forward->N_key);
                    if (idx == 0 && prev_[level]->forward->keys[0] > prev_[level-1]->forward->keys[0]) idx = -1;

                    if (prev_[level]->forward->keys[idx] == prev_[level-1]->forward->keys[0]) {
                        break;
                    } else if (prev_[level]->forward->keys[idx+1] == 0) {
                        prev_[level]->forward->keys[idx+1] = prev_[level-1]->forward->keys[0];
                        prev_[level]->forward->next[idx+1] = prev_[level-1]->forward;
                        prev_[level]->forward->N_key++;
                        if (prev_[level]->forward->N_key == ARR_SIZE) {
                            level++;
                            if (cur_height < level) {
                                max_height_++;
                            } 
                        } else break;
                    } else {
                        Key update_key = prev_[level]->forward->keys[0];
                        std::memmove(&prev_[level]->forward->keys[idx+2], &prev_[level]->forward->keys[idx+1], (prev_[level]->forward->N_key - (idx+1)) * sizeof(Key));
                        std::memmove(&prev_[level]->forward->next[idx+2], &prev_[level]->forward->next[idx+1], (prev_[level]->forward->N_key - (idx+1)) * sizeof(Key));
                        shift_count++; // Signal.Jin
                        prev_[level]->forward->keys[idx+1] = prev_[level-1]->forward->keys[0];
                        prev_[level]->forward->next[idx+1] = prev_[level-1]->forward;
                        prev_[level]->forward->N_key++;
                        if (idx == -1) {
                            for (int i = level+1; i < GetMaxHeight(); i++) {
                                if (prev_[i] != nullptr) {
                                    int idx = findMaxLessOrEqualLinear(prev_[i]->forward->keys, update_key, prev_[i]->forward->N_key);
                                    if (prev_[i]->forward->keys[idx] == update_key) {
                                        prev_[i]->forward->keys[idx] = prev_[level-1]->forward->keys[0];
                                    }
                                    if (idx != 0) break;
                                }
                            }
                        }

                        if (prev_[level]->forward->N_key == ARR_SIZE) {
                            level++;
                            if (cur_height < level) {
                                max_height_++;
                            } 
                        } else break;
                    }
                }
            } else { // Case 2-2: Forward node has no room, so we need to make a new node
                if (level == 0) { // Case 2-2-1: Insert into H0
                    Node* add_node = NewNode(key);
                    add_node->forward = prev_[level]->forward;
                    prev_[level]->forward = add_node;
                    break;
                } else { // Case 2-2-2: Not insert into H0
                    if (prev_[level-1] == head_[level-1]) {
                        Node* add_node = NewNode(prev_[level-1]->forward->keys[0]);
                        add_node->forward = prev_[level]->forward;
                        add_node->next[0] = prev_[level-1]->forward;
                        prev_[level]->forward = add_node;
                    } else {
                        Node* add_node = NewNode(prev_[level-1]->keys[0]);
                        add_node->forward = prev_[level]->forward;
                        add_node->next[0] = prev_[level-1];
                        prev_[level]->forward = add_node;
                    }
                    break;
                }
            }
        } else {
            // Case 3: New node must be inserted between nodes or into prev_ node (not head_)
            if (prev_[level]->N_key != ARR_SIZE) { // Case 3-1: prev_ node has a room, so we insert into that node
                if (level == 0) { // Case 3-1-1: Insert into H0
                    for (int shift = 0; shift < prev_[level]->N_key; shift++) {
                        if (prev_[level]->keys[shift] == key) {
                            stop_flag++;
                            break;
                        } else if (prev_[level]->keys[shift] <= key && key < prev_[level]->keys[shift+1]) {
                            for (int do_shift = prev_[level]->N_key-1; do_shift > shift; do_shift--) {
                                prev_[level]->keys[do_shift+1] = prev_[level]->keys[do_shift];
                            }
                            shift_count++; // Signal.Jin
                            prev_[level]->keys[shift+1] = key;
                            prev_[level]->N_key++;
                            if (prev_[level]->N_key == ARR_SIZE && prev_[level+1] == head_[level+1]) {
                                level++;
                                if (cur_height < level) {
                                    max_height_++;
                                }
                            } else stop_flag++;
                            break;
                        } else if (prev_[level]->keys[shift+1] == 0) {
                            prev_[level]->keys[shift+1] = key;
                            prev_[level]->N_key++;
                            if (prev_[level]->N_key == ARR_SIZE && prev_[level+1] == head_[level+1]) {
                                level++;
                                if (cur_height < level) {
                                    max_height_++;
                                }
                            } else stop_flag++;
                            break;
                        }
                    }
                } else { // Case 3-1-2: Not insert into H0
                    for (int shift = 0; shift < prev_[level]->N_key; shift++) {
                        if (prev_[level]->keys[shift] == prev_[level-1]->keys[0]) {
                            stop_flag++;
                            break;
                        } else if (prev_[level]->keys[shift] < prev_[level-1]->keys[0] && prev_[level-1]->keys[0] < prev_[level]->keys[shift+1]) {
                            for (int do_shift = prev_[level]->N_key-1; do_shift > shift; do_shift--) {
                                prev_[level]->keys[do_shift+1] = prev_[level]->keys[do_shift];
                                prev_[level]->next[do_shift+1] = prev_[level]->next[do_shift];
                            }
                            shift_count++; // Signal.Jin
                            prev_[level]->keys[shift+1] = prev_[level-1]->keys[0];
                            prev_[level]->next[shift+1] = prev_[level-1];
                            prev_[level]->N_key++;
                            if (prev_[level]->N_key == ARR_SIZE && prev_[level+1] == head_[level+1]) {
                                level++;
                                if (cur_height < level) {
                                    max_height_++;
                                }
                            } else stop_flag++;
                            break;
                        } else if (prev_[level]->keys[shift+1] == 0) {
                            prev_[level]->keys[shift+1] = prev_[level-1]->keys[0];
                            prev_[level]->next[shift+1] = prev_[level-1];
                            prev_[level]->N_key++;
                            if (prev_[level]->N_key == ARR_SIZE && prev_[level+1] == head_[level+1]) {
                                level++;
                                if (cur_height < level) {
                                    max_height_++;
                                }
                            } else stop_flag++;
                            break;
                        }
                    }
                }
            } else { // Case 3-2: prev_ node has no room, so we need to make a new node
                // When creating a new node, change prev to the newly created node.
                if (level == 0) { // Case 3-2-1: Insert into H0
                    int no_room = 0;
                    if (prev_[level]->keys[ARR_SIZE-1] < key && prev_[level]->keys[ARR_SIZE-1] != 0) {
                        no_room++;
                    } else {
                        for (int shift = 0; shift < prev_[level]->N_key-1; shift++) {
                            if (prev_[level]->keys[shift] <= key && key < prev_[level]->keys[shift+1]) {
                                Key temp_key = prev_[level]->keys[ARR_SIZE-1];
                                for (int do_shift = prev_[level]->N_key-2; do_shift > shift; do_shift--) {
                                    prev_[level]->keys[do_shift+1] = prev_[level]->keys[do_shift];
                                }
                                shift_count++; // Signal.Jin
                                prev_[level]->keys[shift+1] = key;
                                if (prev_[level]->forward != nullptr && prev_[level]->forward->N_key < ARR_SIZE) {
                                    Key update_key = prev_[level]->forward->keys[0];
                                    for (int d_shift = prev_[level]->forward->N_key; d_shift > 0; d_shift--) {
                                        prev_[level]->forward->keys[d_shift] = prev_[level]->forward->keys[d_shift-1];
                                    }
                                    shift_count++; // Signal.Jin
                                    prev_[level]->forward->keys[0] = temp_key;
                                    prev_[level]->forward->N_key++;
                                    if (false) {
                                        level++;
                                        if (cur_height < level) {
                                            max_height_++;
                                        } 
                                    } else {
                                        for (int update = level+1; update < GetMaxHeight(); update++) {
                                            int loop = 0;
                                            for (loop = 0; loop < prev_[update]->N_key; loop++) {
                                                if (prev_[update]->keys[loop] == update_key) {
                                                    prev_[update]->keys[loop] = temp_key;
                                                    break;
                                                }
                                            }
                                            if (prev_[update]->forward != nullptr) {
                                                for (loop = 0; loop < prev_[update]->forward->N_key; loop++) {
                                                    if (prev_[update]->forward->keys[loop] == update_key) {
                                                        prev_[update]->forward->keys[loop] = temp_key;
                                                        break;
                                                    }
                                                }
                                            }
                                            if (loop != 0) break;
                                        }
                                        stop_flag++;
                                    }
                                    break;
                                } else {
                                    Node* add_node = NewNode(temp_key);
                                    add_node->forward = prev_[level]->forward;
                                    prev_[level]->forward = add_node;
                                    prev_[level] = add_node;
                                    level++;
                                    if (cur_height < level) {
                                        max_height_++;
                                    }
                                    break;
                                }
                            }
                        }
                    }
                    if (no_room != 0) { 
                        if (prev_[level]->forward != nullptr && prev_[level]->forward->N_key < ARR_SIZE) {
                            Key update_key = prev_[level]->forward->keys[0];
                            for (int d_shift = prev_[level]->forward->N_key; d_shift > 0; d_shift--) {
                                prev_[level]->forward->keys[d_shift] = prev_[level]->forward->keys[d_shift-1];
                            }
                            shift_count++; // Signal.Jin
                            prev_[level]->forward->keys[0] = key;
                            prev_[level]->forward->N_key++;
                            if (false) {
                                level++;
                                if (cur_height < level) {
                                    max_height_++;
                                } 
                            } else {
                                for (int update = level+1; update < GetMaxHeight(); update++) {
                                    int loop = 0;
                                    for (loop = 0; loop < prev_[update]->N_key; loop++) {
                                        if (prev_[update]->keys[loop] == update_key) {
                                            prev_[update]->keys[loop] = key;
                                            break;
                                        }
                                    }
                                    if (prev_[update]->forward != nullptr) {
                                        for (loop = 0; loop < prev_[update]->forward->N_key; loop++) {
                                            if (prev_[update]->forward->keys[loop] == update_key) {
                                                prev_[update]->forward->keys[loop] = key;
                                                break;
                                            }
                                        }
                                    }
                                    if (loop != 0) break;
                                }
                                stop_flag++;
                            }
                            break;
                        } else {
                            Node* add_node = NewNode(key);
                            add_node->forward = prev_[level]->forward;
                            prev_[level]->forward = add_node;
                            prev_[level] = add_node;
                            level++;
                            if (cur_height < level) {
                                max_height_++;
                            }
                        }
                    }
                } else { // Case 3-2-2: Not insert into H0
                    int no_room = 0;
                    if (prev_[level]->keys[ARR_SIZE-1] < prev_[level-1]->keys[0] && prev_[level]->keys[ARR_SIZE-1] != 0) {
                        no_room++;
                    } else {
                        for (int shift = 0; shift < prev_[level]->N_key-1; shift++) {
                            if (prev_[level]->keys[shift] <= prev_[level-1]->keys[0] && prev_[level-1]->keys[0] < prev_[level]->keys[shift+1]) {
                                Key temp_key = prev_[level]->keys[ARR_SIZE-1];
                                Node* temp_next = prev_[level]->next[ARR_SIZE-1];
                                for (int do_shift = prev_[level]->N_key-2; do_shift > shift; do_shift--) {
                                    prev_[level]->keys[do_shift+1] = prev_[level]->keys[do_shift];
                                    prev_[level]->next[do_shift+1] = prev_[level]->next[do_shift];
                                }
                                prev_[level]->keys[shift+1] = prev_[level-1]->keys[0];
                                prev_[level]->next[shift+1] = prev_[level-1];
                                if (prev_[level]->forward != nullptr && prev_[level]->forward->N_key < ARR_SIZE) {
                                    Key update_key = prev_[level]->forward->keys[0];
                                    for (int d_shift = prev_[level]->forward->N_key; d_shift > 0; d_shift--) {
                                        prev_[level]->forward->keys[d_shift] = prev_[level]->forward->keys[d_shift-1];
                                        prev_[level]->forward->next[d_shift] = prev_[level]->forward->next[d_shift-1];
                                    }
                                    shift_count++; // Signal.Jin
                                    prev_[level]->forward->keys[0] = temp_key;
                                    prev_[level]->forward->next[0] = temp_next;
                                    prev_[level]->forward->N_key++;
                                    if (false) {
                                        level++;
                                        if (cur_height < level) {
                                            max_height_++;
                                        } 
                                    } else {
                                        for (int update = level+1; update < GetMaxHeight(); update++) {
                                            int loop = 0;
                                            for (loop = 0; loop < prev_[update]->N_key; loop++) {
                                                if (prev_[update]->keys[loop] == update_key) {
                                                    prev_[update]->keys[loop] = temp_key;
                                                    break;
                                                }
                                            }
                                            if (prev_[update]->forward != nullptr) {
                                                for (loop = 0; loop < prev_[update]->forward->N_key; loop++) {
                                                    if (prev_[update]->forward->keys[loop] == update_key) {
                                                        prev_[update]->forward->keys[loop] = temp_key;
                                                        break;
                                                    }
                                                }
                                            }
                                            if (loop != 0) break;
                                        }
                                        stop_flag++;
                                    }
                                    break;
                                } else {
                                    Node* add_node = NewNode(temp_key);
                                    add_node->forward = prev_[level]->forward;
                                    add_node->next[0] = temp_next;
                                    prev_[level]->forward = add_node;
                                    prev_[level] = add_node;
                                    level++;
                                    if (cur_height < level) {
                                        max_height_++;
                                    }
                                    break;
                                }
                            }
                        }
                    }
                    if (no_room != 0) {
                        if (prev_[level]->forward != nullptr && prev_[level]->forward->N_key < ARR_SIZE) {
                            Key update_key = prev_[level]->forward->keys[0];
                            for (int d_shift = prev_[level]->forward->N_key; d_shift > 0; d_shift--) {
                                prev_[level]->forward->keys[d_shift] = prev_[level]->forward->keys[d_shift-1];
                                prev_[level]->forward->next[d_shift] = prev_[level]->forward->next[d_shift-1];
                            }
                            shift_count++; // Signal.Jin
                            prev_[level]->forward->keys[0] = prev_[level-1]->keys[0];
                            prev_[level]->forward->next[0] = prev_[level-1];
                            prev_[level]->forward->N_key++;
                            if (false) {
                                level++;
                                if (cur_height < level) {
                                    max_height_++;
                                } 
                            } else {
                                for (int update = level+1; update < GetMaxHeight(); update++) {
                                    int loop = 0;
                                    for (loop = 0; loop < prev_[update]->N_key; loop++) {
                                        if (prev_[update]->keys[loop] == update_key) {
                                            prev_[update]->keys[loop] = prev_[level]->forward->keys[0];
                                            break;
                                        }
                                    }
                                    if (prev_[update]->forward != nullptr) {
                                        for (loop = 0; loop < prev_[update]->forward->N_key; loop++) {
                                            if (prev_[update]->forward->keys[loop] == update_key) {
                                                prev_[update]->forward->keys[loop] = prev_[level]->forward->keys[0];
                                                break;
                                            }
                                        }
                                    }
                                    if (loop != 0) break;
                                }
                                stop_flag++;
                            }
                            break;
                        } else {
                            Node* add_node = NewNode(prev_[level-1]->keys[0]);
                            add_node->forward = prev_[level]->forward;
                            add_node->next[0] = prev_[level-1];
                            prev_[level]->forward = add_node;
                            prev_[level] = add_node;
                            level++;
                            if (cur_height < level) {
                                max_height_++;
                            }
                        }
                    }    
                }
            }
        }

        if (stop_flag > 0) {
            break;
        }
    }
}

template<typename Key>
void SkipList<Key>::Insert_Array(const Key& key) {
    Node* prev_[MAXHEIGHT];
    std::copy(std::begin(head_), std::end(head_), std::begin(prev_));
    int height = GetMaxHeight() - 1;
    Node* x = head_[height];

    while (true) {
        while (x->forward != nullptr && compare_(x->forward->keys[0], key) <= 0) {
            if (compare_(x->forward->keys[0], key) == 0) {
                return;
            } else {
                x = x->forward;
            }
        }
        prev_[height] = x;
        height--;
        if (height >= 0) {
            int n_key = x->N_key;
            x = (x == head_[height + 1]) ? head_[height] : x->next[findMaxLessOrEqualLinear(x->keys, key, n_key)];
        } else {
            break;
        }
    }

    if (prev_[0] != head_[0]) {
        for (int i = 0; i < prev_[0]->N_key; i++) {
            if (compare_(prev_[0]->keys[i], key) == 0) {
                return;
            }
        }
    }

    int node_height = randomLevel();
    int level = 0;

    while (level <= node_height) {
        if (level == 0) {
            if (prev_[level] == head_[level] && prev_[level]->forward == nullptr) {
                Node* fnode = NewNode(key);
                fnode->forward = prev_[level]->forward;
                prev_[level]->forward = fnode;
                prev_[level] = fnode;
            } else if (prev_[level] == head_[level] && prev_[level]->forward != nullptr) {
                if (prev_[level]->forward->N_key == ARR_SIZE) {
                    Node* nnode = NewNode(key);
                    nnode->forward = prev_[level]->forward;
                    prev_[level]->forward = nnode;
                    prev_[level] = nnode;
                } else {
                    for (int do_shift = prev_[level]->forward->N_key-1; do_shift >= 0; do_shift--) {
                        prev_[level]->forward->keys[do_shift+1] = prev_[level]->forward->keys[do_shift];
                    }
                    prev_[level]->forward->keys[0] = key;
                    prev_[level]->forward->N_key++;
                }
            } else {
                if (prev_[level]->N_key == ARR_SIZE) {
                    if (prev_[level]->keys[ARR_SIZE-1] < key) {
                        if (prev_[level]->forward == nullptr || prev_[level]->forward->N_key == ARR_SIZE) {
                            Node* nnode = NewNode(key);
                            nnode->forward = prev_[level]->forward;
                            prev_[level]->forward = nnode;
                            prev_[level] = nnode;
                        } else if (prev_[level]->forward->N_key < ARR_SIZE) {
                            for (int do_shift = prev_[level]->forward->N_key-1; do_shift >= 0; do_shift--) {
                                prev_[level]->forward->keys[do_shift+1] = prev_[level]->forward->keys[do_shift];
                            }
                            prev_[level]->forward->keys[0] = key;
                            prev_[level]->forward->N_key++;
                        }
                    } else {
                        Key temp_key = prev_[level]->keys[ARR_SIZE-1];
                        for (int shift = 0; shift < ARR_SIZE-1; shift++) {
                            if (prev_[level]->keys[shift] <= key && key < prev_[level]->keys[shift+1]) {
                                for (int do_shift = prev_[level]->N_key-2; do_shift > shift; do_shift--) {
                                    prev_[level]->keys[do_shift+1] = prev_[level]->keys[do_shift];
                                }
                                prev_[level]->keys[shift+1] = key;
                                break;
                            }
                        }
                        if (prev_[level]->forward == nullptr || prev_[level]->forward->N_key == ARR_SIZE) {
                            Node* nnode = NewNode(temp_key);
                            nnode->forward = prev_[level]->forward;
                            prev_[level]->forward = nnode;
                        } else if (prev_[level]->forward->N_key < ARR_SIZE) {
                            for (int do_shift = prev_[level]->forward->N_key-1; do_shift >= 0; do_shift--) {
                                prev_[level]->forward->keys[do_shift+1] = prev_[level]->forward->keys[do_shift];
                            }
                            prev_[level]->forward->keys[0] = temp_key;
                            prev_[level]->forward->N_key++;
                        }
                    }
                } else if (prev_[level]->N_key < ARR_SIZE) {
                    for (int shift = 0; shift < prev_[level]->N_key; shift++) {
                        if (prev_[level]->keys[shift] <= key && key < prev_[level]->keys[shift+1]) {
                            for (int do_shift = prev_[level]->N_key-1; do_shift > shift; do_shift--) {
                                prev_[level]->keys[do_shift+1] = prev_[level]->keys[do_shift];
                            }
                            prev_[level]->keys[shift+1] = key;
                            prev_[level]->N_key++;
                            break;
                        } else if (prev_[level]->keys[shift+1] == 0) {
                            prev_[level]->keys[shift+1] = key;
                            prev_[level]->N_key++;
                            break;
                        }
                    }
                }
            }
        } else if (level > 0) { // Not for level (0)
            if (prev_[level] == head_[level] && prev_[level]->forward == nullptr) {
                Node* fnode = NewNode(key);
                fnode->forward = prev_[level]->forward;
                fnode->next[0] = prev_[level-1];
                prev_[level]->forward = fnode;
                prev_[level] = fnode;
            } else if (prev_[level] == head_[level] && prev_[level]->forward != nullptr) {
                if (prev_[level]->forward->N_key == ARR_SIZE) {
                    Node* nnode = NewNode(key);
                    nnode->forward = prev_[level]->forward;
                    nnode->next[0] = prev_[level-1];
                    prev_[level]->forward = nnode;
                    prev_[level] = nnode;
                } else {
                    for (int do_shift = prev_[level]->forward->N_key-1; do_shift >= 0; do_shift--) {
                        prev_[level]->forward->keys[do_shift+1] = prev_[level]->forward->keys[do_shift];
                        prev_[level]->forward->next[do_shift+1] = prev_[level]->forward->next[do_shift];
                    }
                    prev_[level]->forward->keys[0] = key;
                    prev_[level]->forward->next[0] = prev_[level-1];
                    prev_[level]->forward->N_key++;
                }
            } else {
                if (prev_[level]->N_key == ARR_SIZE) {
                    if (prev_[level]->keys[ARR_SIZE-1] < key) {
                        if (prev_[level]->forward == nullptr || prev_[level]->forward->N_key == ARR_SIZE) {
                            Node* nnode = NewNode(key);
                            nnode->forward = prev_[level]->forward;
                            nnode->next[0] = prev_[level-1];
                            prev_[level]->forward = nnode;
                            prev_[level] = nnode;
                        } else if (prev_[level]->forward->N_key < ARR_SIZE) {
                            for (int do_shift = prev_[level]->forward->N_key-1; do_shift >= 0; do_shift--) {
                                prev_[level]->forward->keys[do_shift+1] = prev_[level]->forward->keys[do_shift];
                                prev_[level]->forward->next[do_shift+1] = prev_[level]->forward->next[do_shift];
                            }
                            prev_[level]->forward->keys[0] = key;
                            prev_[level]->forward->next[0] = prev_[level-1];
                            prev_[level]->forward->N_key++;
                        }
                    } else {
                        Key temp_key = prev_[level]->keys[ARR_SIZE-1];
                        Node* temp_next = prev_[level]->next[ARR_SIZE-1];
                        for (int shift = 0; shift < ARR_SIZE-1; shift++) {
                            if (prev_[level]->keys[shift] <= key && key < prev_[level]->keys[shift+1]) {
                                for (int do_shift = prev_[level]->N_key-2; do_shift > shift; do_shift--) {
                                    prev_[level]->keys[do_shift+1] = prev_[level]->keys[do_shift];
                                    prev_[level]->next[do_shift+1] = prev_[level]->next[do_shift];
                                }
                                prev_[level]->keys[shift+1] = key;
                                prev_[level]->next[shift+1] = prev_[level-1];
                                break;
                            }
                        }
                        if (prev_[level]->forward == nullptr || prev_[level]->forward->N_key == ARR_SIZE) {
                            Node* nnode = NewNode(temp_key);
                            nnode->forward = prev_[level]->forward;
                            nnode->next[0] = temp_next;
                            prev_[level]->forward = nnode;
                        } else if (prev_[level]->forward->N_key < ARR_SIZE) {
                            for (int do_shift = prev_[level]->forward->N_key-1; do_shift >= 0; do_shift--) {
                                prev_[level]->forward->keys[do_shift+1] = prev_[level]->forward->keys[do_shift];
                                prev_[level]->forward->next[do_shift+1] = prev_[level]->forward->next[do_shift];
                            }
                            prev_[level]->forward->keys[0] = temp_key;
                            prev_[level]->forward->next[0] = temp_next;
                            prev_[level]->forward->N_key++;
                        }
                    }
                } else if (prev_[level]->N_key < ARR_SIZE) {
                    for (int shift = 0; shift < prev_[level]->N_key; shift++) {
                        if (prev_[level]->keys[shift] <= key && key < prev_[level]->keys[shift+1]) {
                            for (int do_shift = prev_[level]->N_key-1; do_shift > shift; do_shift--) {
                                prev_[level]->keys[do_shift+1] = prev_[level]->keys[do_shift];
                                prev_[level]->next[do_shift+1] = prev_[level]->next[do_shift];
                            }
                            prev_[level]->keys[shift+1] = key;
                            prev_[level]->next[shift+1] = prev_[level-1];
                            prev_[level]->N_key++;
                            break;
                        } else if (prev_[level]->keys[shift+1] == 0) {
                            prev_[level]->keys[shift+1] = key;
                            prev_[level]->next[shift+1] = prev_[level-1];
                            prev_[level]->N_key++;
                            break;
                        }
                    }
                }
            }
        }
        level++;
    }
}

template<typename Key>
bool SkipList<Key>::Contains(const Key& key) const {
    int height = GetMaxHeight() - 1;
    Key result_key = -1;
    Node* x = head_[height]; // Use when searching

    if (x->forward != nullptr && compare_(x->forward->keys[0], key) <= 0) x = x->forward;

    while (true) { // Find the location which will insert the key using prev_ and head_
        height--;
        if (height >= 0) {
            int n_key = x->N_key;
            if (n_key <= ARR_SIZE/2) {
                x = (x == head_[height + 1]) ? head_[height] : x->next[findMaxLessOrEqualLinearSIMD(x->keys, key, n_key)];
            } else {
                x = (x == head_[height + 1]) ? head_[height] : x->next[findMaxLessOrEqualBinary(x->keys, key)];
            }
        } else break;
    }

    int idx2 = findMaxLessOrEqualBinary(x->keys, key);
    result_key = x->keys[idx2];

    if (x != nullptr && compare_(result_key, key) == 0) {
        return true;
    } else {
        return false;
    }
}

template<typename Key>
bool SkipList<Key>::Contains_Height(const Key& key) const {
    int height = GetMaxHeight() - 1;
    Key result_key = -1;
    Node* x = head_[height]; // Use when searching

    if (x->forward != nullptr && compare_(x->forward->keys[0], key) <= 0) x = x->forward;

    while (true) { // Find the location which will insert the key using prev_ and head_
        height--;
        if (height >= 0) {
            int n_key = x->N_key;
            x = (x == head_[height + 1]) ? head_[height] : x->next[findMaxLessOrEqualLinear(x->keys, key, n_key)];
        } else break;
    }

    int n_key = x->N_key;
    int idx2 = findMaxLessOrEqualLinear(x->keys, key, n_key);
    result_key = x->keys[idx2];

    if (x != nullptr && compare_(result_key, key) == 0) {
        return true;
    } else {
        return false;
    }
}


template<typename Key>
Key SkipList<Key>::Scan(const Key& key, const int scan_num) {
    int height = GetMaxHeight() - 1;
    int result_key;
    Key temp_key;

    Node* x = head_[height]; // Use when searching

    while (x->forward != nullptr && compare_(x->forward->keys[0], key) <= 0) {
        x = x->forward;
    }

    while (true) { // Find the location which will insert the key using prev_ and head_
        height--;
        if (height >= 0) {
            int n_key = x->N_key;
            if (n_key <= ARR_SIZE/2) {
                x = (x == head_[height + 1]) ? head_[height] : x->next[findMaxLessOrEqualLinear(x->keys, key, n_key)];
            } else {
                x = (x == head_[height + 1]) ? head_[height] : x->next[findMaxLessOrEqualBinary(x->keys, key)];
            }
        } else break;
    }

    // Find key from entry(Array)
    if (compare_(x->keys[0], key) == 0) {
        result_key = 0;
    } else {
        int idx2 = findMaxLessOrEqual(x->keys, key);
        result_key = idx2;
    }
    int i = 0;
    while (true) {
        if (result_key != 0) {
            for (int j = result_key; j < ARR_SIZE; j++) {
                temp_key = x->keys[j];
                i++;
            }
            result_key = 0;
        } else {
            for (int k = 0; k < ARR_SIZE; k++) {
                temp_key = x->keys[k];
                i++;
            }
        }
        if (i >= scan_num-1) {
            break;
        }
        if (x->forward != nullptr) {
            x = x->forward;
        } else {
            break;
        }
    }
    return temp_key;
}

template<typename Key>
void SkipList<Key>::Print() const {
    printf("Print height - %d\n", GetMaxHeight());
    for(int i = 0; i < GetMaxHeight(); i++) {
        if (head_[i]->forward != nullptr) {
            Node* x = head_[i];
            while (x->forward != nullptr) {
                x = x->forward;
                for (int i = 0; i < x->N_key; i++) {
                    std::cout << x->keys[i] << " ";
                }
                std::cout << std::endl;
            }
        }
        printf("Next\n");
    }
}

template<typename Key>
void SkipList<Key>::Array_utilization() {
    int m_height = GetMaxHeight() - 1;
    int cur_height = 0;

    while (cur_height <= m_height) {
        printf("Current Layer = %d\n", cur_height);
        Node* x = head_[cur_height];
        while(true) {
            if (x->forward == nullptr) break;
            x = x->forward;
            //printf("here\n");
            
            double util = 0;
            util = ((double)x->N_key / (double)ARR_SIZE) * 100;
            printf("%.2f\n", util);
        }
        cur_height++;
    }
}

template<typename Key>
void SkipList<Key>::Insert_future(const Key& key) {
    Node* prev_[MAXHEIGHT];
    std::copy(std::begin(head_), std::end(head_), std::begin(prev_));
    int height = GetMaxHeight() - 1; // Using for search
    Node* x = head_[height]; // Use when searching

    if (x->forward != nullptr && compare_(x->forward->keys[0], key) <= 0) x = x->forward;

    while (true) { // Find the location which will insert the key using prev_ and head_
        prev_[height--] = x;
        //printf("lookup prev key = %lu\n", prev_[height+1]->keys[0]);
        if (height >= 0) {
            int n_key = x->N_key;
            if (n_key <= ARR_SIZE/2) {
                x = (x == head_[height + 1]) ? head_[height] : x->next[findMaxLessOrEqualLinear(x->keys, key, n_key)];
            } else {
                x = (x == head_[height + 1]) ? head_[height] : x->next[findMaxLessOrEqualBinary(x->keys, key)];
            }
        } else {
            break;
        }
    }
    // Do not allow duplicated key
    if (prev_[0] != head_[0]) {
        for (int i = 0; i < prev_[0]->N_key; i++) {
            if (compare_(prev_[0]->keys[i], key) == 0) {
                return;
            }
        }
    }
    
    // Do insert operation
    int level = 0;
    while (true) {
        int stop_flag = 0;
        int cur_height = GetMaxHeight() - 1;
        if (prev_[level] == head_[level] && prev_[level]->forward == nullptr) {
        // Case 1: There is no node in list and need to make a new node
            if (level == 0) { // Case 1-1: If insert into H0
                Node* Elist_node = NewNode(key);
                Elist_node->forward = prev_[level]->forward;
                prev_[level]->forward = Elist_node;
                break;
            } else { // Case 1-2: If not insert into H0
                if (cur_height < level) {
                    max_height_++;
                }
                if (prev_[level-1]->forward != nullptr && prev_[level-1] == head_[level-1]) {
                    Node* Elist_node = NewNode(prev_[level-1]->forward->keys[0]);
                    Elist_node->forward = prev_[level]->forward;
                    Elist_node->next[0] = prev_[level-1]->forward;
                    prev_[level]->forward = Elist_node;
                } else {
                    Node* Elist_node = NewNode(prev_[level-1]->keys[0]);
                    Elist_node->forward = prev_[level]->forward;
                    Elist_node->next[0] = prev_[level-1];
                    prev_[level]->forward = Elist_node;
                }
                break;
            }
        } else if (prev_[level] == head_[level]) {
            // Case 2: There is another node in list and need to make a new node
            if (prev_[level]->forward->N_key != ARR_SIZE) { // Case 2-1: Forward node has a room
                if (level == 0) { // Case 2-1-1: Insert into H0
                    int idx = findMaxLessOrEqual(prev_[level]->forward->keys, key);
                    if (idx == 0 && prev_[level]->forward->keys[0] > key) idx = -1;

                    if (prev_[level]->forward->keys[idx] == key) {
                        break;
                    } else if (prev_[level]->forward->keys[idx+1] == 0) {
                        prev_[level]->forward->keys[idx+1] = key;
                        prev_[level]->forward->N_key++;
                        if (prev_[level]->forward->N_key == ARR_SIZE) {
                            level++;
                            if (cur_height < level) {
                                max_height_++;
                            } 
                        } else break;
                    } else {
                        Key update_key = prev_[level]->forward->keys[0];
                        std::memmove(&prev_[level]->forward->keys[idx+2], &prev_[level]->forward->keys[idx+1], (prev_[level]->forward->N_key - (idx+1)) * sizeof(Key));
                        shift_count++; // Signal.Jin
                        prev_[level]->forward->keys[idx+1] = key;
                        prev_[level]->forward->N_key++;
                        if (idx == -1) {
                            for (int i = level+1; i < GetMaxHeight(); i++) {
                                if (prev_[i] != nullptr) {
                                    int idx = findMaxLessOrEqual(prev_[i]->forward->keys, update_key);
                                    if (prev_[i]->forward->keys[idx] == update_key) {
                                        prev_[i]->forward->keys[idx] = key;
                                    }
                                    if (idx != 0) break;
                                }
                            }
                        }

                        if (prev_[level]->forward->N_key == ARR_SIZE) {
                            level++;
                            if (cur_height < level) {
                                max_height_++;
                            } 
                        } else break;
                    }
                } else { // Case 2-1-2: Not insert into H0
                    int idx = findMaxLessOrEqual(prev_[level]->forward->keys, prev_[level-1]->forward->keys[0]);
                    if (idx == 0 && prev_[level]->forward->keys[0] > prev_[level-1]->forward->keys[0]) idx = -1;

                    if (prev_[level]->forward->keys[idx] == prev_[level-1]->forward->keys[0]) {
                        break;
                    } else if (prev_[level]->forward->keys[idx+1] == 0) {
                        prev_[level]->forward->keys[idx+1] = prev_[level-1]->forward->keys[0];
                        prev_[level]->forward->next[idx+1] = prev_[level-1]->forward;
                        prev_[level]->forward->N_key++;
                        if (prev_[level]->forward->N_key == ARR_SIZE) {
                            level++;
                            if (cur_height < level) {
                                max_height_++;
                            } 
                        } else break;
                    } else {
                        Key update_key = prev_[level]->forward->keys[0];
                        std::memmove(&prev_[level]->forward->keys[idx+2], &prev_[level]->forward->keys[idx+1], (prev_[level]->forward->N_key - (idx+1)) * sizeof(Key));
                        std::memmove(&prev_[level]->forward->next[idx+2], &prev_[level]->forward->next[idx+1], (prev_[level]->forward->N_key - (idx+1)) * sizeof(Key));
                        shift_count++; // Signal.Jin
                        prev_[level]->forward->keys[idx+1] = prev_[level-1]->forward->keys[0];
                        prev_[level]->forward->next[idx+1] = prev_[level-1]->forward;
                        prev_[level]->forward->N_key++;
                        if (idx == -1) {
                            for (int i = level+1; i < GetMaxHeight(); i++) {
                                if (prev_[i] != nullptr) {
                                    int idx = findMaxLessOrEqual(prev_[i]->forward->keys, update_key);
                                    if (prev_[i]->forward->keys[idx] == update_key) {
                                        prev_[i]->forward->keys[idx] = prev_[level-1]->forward->keys[0];
                                    }
                                    if (idx != 0) break;
                                }
                            }
                        }

                        if (prev_[level]->forward->N_key == ARR_SIZE) {
                            level++;
                            if (cur_height < level) {
                                max_height_++;
                            } 
                        } else break;
                    }
                }
            } else { // Case 2-2: Forward node has no room, so we need to make a new node
                if (level == 0) { // Case 2-2-1: Insert into H0
                    int idx = findMaxLessOrEqual(prev_[level]->forward->keys, key);
                    if (idx == 0 && prev_[level]->forward->keys[0] > key) {
                        Node* add_node = NewNode(key);
                        add_node->forward = prev_[level]->forward;
                        prev_[level]->forward = add_node;    
                    } else {
                        Node* add_node = NewNode(key);
                        std::memcpy(add_node->keys+1, &prev_[level]->forward->keys[idx+1], (ARR_SIZE - (idx+1)) * sizeof(Key));
                        std::memset(&prev_[level]->forward->keys[idx+1], 0, (ARR_SIZE - (idx+1)) * sizeof(Key));
                        shift_count++; // Signal.Jin
                        add_node->N_key += ARR_SIZE - (idx+1);
                        add_node->forward = prev_[level]->forward->forward;
                        prev_[level]->forward->N_key -= ARR_SIZE - (idx+1);
                        prev_[level]->forward->forward = add_node;
                        prev_[level] = add_node;
                        level++; // Keep tracking
                        if (cur_height < level) {
                            max_height_++;
                        }
                    }
                } else { // Case 2-2-2: Not insert into H0
                    if (prev_[level-1] == head_[level-1]) {
                        Node* add_node = NewNode(prev_[level-1]->forward->keys[0]);
                        add_node->forward = prev_[level]->forward;
                        add_node->next[0] = prev_[level-1]->forward;
                        prev_[level]->forward = add_node;
                    } else {
                        Node* add_node = NewNode(prev_[level-1]->keys[0]);
                        add_node->forward = prev_[level]->forward;
                        add_node->next[0] = prev_[level-1];
                        prev_[level]->forward = add_node;
                    }
                    break;
                }
            }
        } else {
            // Case 3: New node must be inserted between nodes or into prev_ node (not head_)
            if (prev_[level]->N_key != ARR_SIZE) { // Case 3-1: prev_ node has a room, so we insert into that node
                if (level == 0) { // Case 3-1-1: Insert into H0
                    int idx = findMaxLessOrEqual(prev_[level]->keys, key);
                    if (idx == 0 && prev_[level]->keys[0] > key) idx = -1;
                    //printf("Case 3-1 Index = %d\n", idx);
                    if (prev_[level]->keys[idx] == key) {
                        stop_flag++;
                    } else {
                        if (prev_[level]->keys[idx+1] == 0) {
                            prev_[level]->keys[idx+1] = key;
                            prev_[level]->N_key++;    
                        } else {
                            std::memmove(&prev_[level]->keys[idx+2], &prev_[level]->keys[idx+1], (prev_[level]->N_key - (idx+1)) * sizeof(Key));
                            shift_count++; // Signal.Jin
                            prev_[level]->keys[idx+1] = key;
                            prev_[level]->N_key++;
                        }
                        if (prev_[level]->N_key == ARR_SIZE && prev_[level+1] == head_[level+1]) {
                            level++;
                            if (cur_height < level) {
                                max_height_++;
                            }
                        } else stop_flag++;
                    }
                } else { // Case 3-1-2: Not insert into H0
                    int idx = findMaxLessOrEqual(prev_[level]->keys, prev_[level-1]->keys[0]);
                    if (idx == 0 && prev_[level]->keys[0] > prev_[level-1]->keys[0]) idx = -1;
                    //printf("Case 3-1-2 Index = %d\n", idx);
                    if (prev_[level]->keys[idx] == prev_[level-1]->keys[0]) {
                        stop_flag++;
                    } else {
                        if (prev_[level]->keys[idx+1] == 0) {
                            prev_[level]->keys[idx+1] = prev_[level-1]->keys[0];
                            prev_[level]->next[idx+1] = prev_[level-1];
                            prev_[level]->N_key++;
                        } else {
                            std::memmove(&prev_[level]->keys[idx+2], &prev_[level]->keys[idx+1], (prev_[level]->N_key - (idx+1)) * sizeof(Key));
                            std::memmove(&prev_[level]->next[idx+2], &prev_[level]->next[idx+1], (prev_[level]->N_key - (idx+1)) * sizeof(Key));
                            shift_count++; // Signal.Jin
                            prev_[level]->keys[idx+1] = prev_[level-1]->keys[0];
                            prev_[level]->next[idx+1] = prev_[level-1];
                            prev_[level]->N_key++;
                        }
                    }
                    if (prev_[level]->N_key == ARR_SIZE && prev_[level+1] == head_[level+1]) {
                        level++;
                        if (cur_height < level) {
                            max_height_++;
                        }
                    } else stop_flag++;
                }
            } else { // Case 3-2: prev_ node has no room, so we need to make a new node
                // When creating a new node, change prev to the newly created node.
                // Uneven-split operaiton

                //auto sp_start = Clock::now();
                if (level == 0) { // Case 3-2-1: Insert into H0
                    split_count++; // Signal.Jin
                    int idx = findMaxLessOrEqual(prev_[level]->keys, key);
                    if (idx == 0 && prev_[level]->keys[0] > key) idx = -1;
                    { 
                        if (idx == ARR_SIZE-1) {
                            if (prev_[level]->forward != nullptr && prev_[level]->forward->N_key < ARR_SIZE) {
                                Key update_key = prev_[level]->forward->keys[0];
                                std::memmove(&prev_[level]->forward->keys[1], &prev_[level]->forward->keys[0], (prev_[level]->forward->N_key) * sizeof(Key));
                                prev_[level]->forward->keys[0] = key;
                                prev_[level]->forward->N_key++;
                                for (int i = level+1; i < GetMaxHeight(); i++) {
                                    if (prev_[i] != nullptr) {
                                        int idx = findMaxLessOrEqual(prev_[i]->keys, update_key);
                                        if (prev_[i]->keys[idx] == update_key) {
                                            prev_[i]->keys[idx] = key;
                                            if (idx != 0) break;
                                        } else {
                                            if (prev_[i]->forward != nullptr) {
                                                int idx2 = findMaxLessOrEqual(prev_[i]->forward->keys, update_key);
                                                if (prev_[i]->forward->keys[idx2] == update_key) {
                                                    prev_[i]->forward->keys[idx2] = key;
                                                    if (idx2 != 0) break;
                                                }
                                            }
                                        }
                                    }
                                }
                                break;
                            } else {
                                Node* add_node = NewNode(key);
                                add_node->forward = prev_[level]->forward;
                                prev_[level]->forward = add_node;
                                prev_[level] = add_node;
                                level++;
                                if (cur_height < level) {
                                    max_height_++;
                                }
                            }
                        } else {
                            if (idx < ARR_SIZE / 2) {
                                Node* add_node = NewNode(prev_[level]->keys[ARR_SIZE/2]);
                                std::memcpy(add_node->keys, &prev_[level]->keys[ARR_SIZE/2], (ARR_SIZE/2) * sizeof(Key));
                                std::memset(&prev_[level]->keys[ARR_SIZE/2], 0, (ARR_SIZE/2) * sizeof(Key));
                                shift_count++; // Signal.Jin
                                add_node->N_key = ARR_SIZE/2;
                                add_node->forward = prev_[level]->forward;
                                prev_[level]->forward = add_node;
                                if (idx == ARR_SIZE / 2 - 1) {
                                    prev_[level]->keys[idx + 1] = key;
                                    prev_[level]->N_key = ARR_SIZE / 2 + 1;
                                } else {
                                    Key update_key = prev_[level]->keys[0];
                                    prev_[level]->N_key = ARR_SIZE / 2;
                                    std::memmove(&prev_[level]->keys[idx+2], &prev_[level]->keys[idx+1], (prev_[level]->N_key - (idx+1)) * sizeof(Key));
                                    shift_count++; // Signal.Jin
                                    prev_[level]->keys[idx+1] = key;
                                    prev_[level]->N_key++;
                                    if (idx == -1) {
                                        for (int update = level+1; update < GetMaxHeight(); update++) {
                                            int idx = findMaxLessOrEqual(prev_[update]->forward->keys, update_key);
                                            //printf("Case 3-2-1 (2) Index = %d\n", idx);
                                            if (prev_[update]->forward->keys[idx] == update_key) {
                                                prev_[update]->forward->keys[idx] = key;
                                            }
                                            if (idx != 0) break;
                                        }
                                        stop_flag++;
                                    }
                                }
                                prev_[level] = add_node;
                                level++; // Keep tracking
                                if (cur_height < level) {
                                    max_height_++;
                                }
                            } else {
                                prev_[level]->N_key = ARR_SIZE / 2;
                                idx = idx - ARR_SIZE / 2;
                                Node* add_node = NewNode(prev_[level]->keys[ARR_SIZE/2]);
                                std::memcpy(add_node->keys, &prev_[level]->keys[ARR_SIZE/2], (ARR_SIZE/2) * sizeof(Key));
                                std::memset(&prev_[level]->keys[ARR_SIZE/2], 0, (ARR_SIZE/2) * sizeof(Key));
                                shift_count++; // Signal.Jin
                                add_node->N_key = ARR_SIZE/2 + 1;
                                add_node->forward = prev_[level]->forward;
                                prev_[level]->forward = add_node;
                                prev_[level] = add_node;
                                std::memmove(&prev_[level]->keys[idx+2], &prev_[level]->keys[idx+1], (prev_[level]->N_key - (idx+1)) * sizeof(Key));
                                shift_count++; // Signal.Jin
                                prev_[level]->keys[idx+1] = key;
                                level++;
                                if (cur_height < level) {
                                    max_height_++;
                                }
                            }
                        }
                    }
                } else { // Case 3-2-2: Not insert into H0
                    split_count++; // Signal.Jin
                    int idx = findMaxLessOrEqual(prev_[level]->keys, prev_[level-1]->keys[0]);
                    if (idx == 0 && prev_[level]->keys[0] > prev_[level-1]->keys[0]) idx = -1;
                    {
                        if (idx == ARR_SIZE-1) {
                            if (prev_[level]->forward != nullptr && prev_[level]->forward->N_key < ARR_SIZE) {
                                Key update_key = prev_[level]->forward->keys[0];
                                std::memmove(&prev_[level]->forward->keys[1], &prev_[level]->forward->keys[0], (prev_[level]->forward->N_key) * sizeof(Key));
                                std::memmove(&prev_[level]->forward->next[1], &prev_[level]->forward->next[0], (prev_[level]->forward->N_key) * sizeof(Key));
                                prev_[level]->forward->keys[0] = prev_[level-1]->keys[0];
                                prev_[level]->forward->next[0] = prev_[level-1];
                                prev_[level]->forward->N_key++;
                                for (int i = level+1; i < GetMaxHeight(); i++) {
                                    if (prev_[i] != nullptr) {
                                        int idx = findMaxLessOrEqual(prev_[i]->keys, update_key);
                                        if (prev_[i]->keys[idx] == update_key) {
                                            prev_[i]->keys[idx] = prev_[level-1]->keys[0];
                                            if (idx != 0) break;
                                        } else {
                                            if (prev_[i]->forward != nullptr) {
                                                int idx2 = findMaxLessOrEqual(prev_[i]->forward->keys, update_key);
                                                if (prev_[i]->forward->keys[idx2] == update_key) {
                                                    prev_[i]->forward->keys[idx2] = prev_[level-1]->keys[0];
                                                    if (idx2 != 0) break;
                                                }
                                            }
                                        }
                                    }
                                }
                                break;
                            } else {
                                Node* add_node = NewNode(prev_[level-1]->keys[0]);
                                add_node->forward = prev_[level]->forward;
                                add_node->next[0] = prev_[level-1];
                                prev_[level]->forward = add_node;
                                prev_[level] = add_node;
                                level++;
                                if (cur_height < level) {
                                    max_height_++;
                                }
                            }
                        } else {
                            if (idx < ARR_SIZE / 2) {
                                Node* add_node = NewNode(prev_[level]->keys[ARR_SIZE/2]);
                                std::memcpy(add_node->keys, &prev_[level]->keys[ARR_SIZE/2], (ARR_SIZE/2) * sizeof(Key));
                                std::memcpy(add_node->next, &prev_[level]->next[ARR_SIZE/2], (ARR_SIZE/2) * sizeof(prev_[level]->next[0]));
                                std::memset(&prev_[level]->keys[ARR_SIZE/2], 0, (ARR_SIZE/2) * sizeof(Key));
                                shift_count++; // Signal.Jin
                                for (int i = ARR_SIZE/2; i < ARR_SIZE; i++) {
                                    prev_[level]->next[i] = nullptr;
                                }
                                add_node->N_key = ARR_SIZE/2;
                                add_node->forward = prev_[level]->forward;
                                prev_[level]->forward = add_node;
                                if (prev_[level]->keys[idx+1] == 0) {
                                    prev_[level]->keys[idx+1] = prev_[level-1]->keys[0];
                                    prev_[level]->next[idx+1] = prev_[level-1];
                                    prev_[level]->N_key = ARR_SIZE / 2 + 1;
                                } else {
                                    Key update_key = prev_[level]->keys[0];
                                    prev_[level]->N_key = ARR_SIZE / 2;
                                    std::memmove(&prev_[level]->keys[idx+2], &prev_[level]->keys[idx+1], (prev_[level]->N_key - (idx+1)) * sizeof(Key));
                                    std::memmove(&prev_[level]->next[idx+2], &prev_[level]->next[idx+1], (prev_[level]->N_key - (idx+1)) * sizeof(prev_[level]->next[0]));
                                    shift_count++; // Signal.Jin
                                    prev_[level]->keys[idx+1] = prev_[level-1]->keys[0];
                                    prev_[level]->next[idx+1] = prev_[level-1];
                                    prev_[level]->N_key++;
                                    if (idx == -1) {
                                        for (int update = level+1; update < GetMaxHeight(); update++) {
                                            int idx = findMaxLessOrEqual(prev_[update]->forward->keys, update_key);
                                            //printf("Case 3-2-2 (2) Index = %d\n", idx);
                                            if (prev_[update]->forward->keys[idx] == update_key) {
                                                prev_[update]->forward->keys[idx] = prev_[level]->forward->keys[0];
                                            }
                                            if (idx != 0) break;
                                        }
                                        stop_flag++;
                                    }
                                }
                                prev_[level] = add_node;
                                level++; // Keep tracking
                                if (cur_height < level) {
                                    max_height_++;
                                }
                            } else {
                                prev_[level]->N_key = ARR_SIZE / 2;
                                idx = idx - ARR_SIZE / 2;
                                Node* add_node = NewNode(prev_[level]->keys[ARR_SIZE/2]);
                                std::memcpy(add_node->keys, &prev_[level]->keys[ARR_SIZE/2], (ARR_SIZE/2) * sizeof(Key));
                                std::memcpy(add_node->next, &prev_[level]->next[ARR_SIZE/2], (ARR_SIZE/2) * sizeof(prev_[level]->next[0]));
                                std::memset(&prev_[level]->keys[ARR_SIZE/2], 0, (ARR_SIZE/2) * sizeof(Key));
                                shift_count++; // Signal.Jin
                                for (int i = ARR_SIZE/2; i < ARR_SIZE; i++) {
                                    prev_[level]->next[i] = nullptr;
                                }
                                add_node->N_key = ARR_SIZE/2 + 1;
                                add_node->forward = prev_[level]->forward;
                                prev_[level]->forward = add_node;
                                prev_[level] = add_node;
                                std::memmove(&prev_[level]->keys[idx+2], &prev_[level]->keys[idx+1], (prev_[level]->N_key - (idx+1)) * sizeof(Key));
                                std::memmove(&prev_[level]->next[idx+2], &prev_[level]->next[idx+1], (prev_[level]->N_key - (idx+1)) * sizeof(prev_[level]->next[0]));
                                shift_count++; // Signal.Jin
                                prev_[level]->keys[idx+1] = prev_[level-1]->keys[0];
                                prev_[level]->next[idx+1] = prev_[level-1];
                                level++;
                                if (cur_height < level) {
                                    max_height_++;
                                }
                            }
                        }
                    }   
                }
                //auto sp_end = Clock::now();
                //split_time += std::chrono::duration_cast<std::chrono::nanoseconds>(sp_end - sp_start).count();
            }
        }

        if (stop_flag > 0) {
            break;
        }
    }
}