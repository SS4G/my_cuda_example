
#include<iostream>

void compress(std::vector<int> &arr) {
    int wr_idx = arr.size() - 1;
    //size_t rd_idx = arr.size() - 1;
    size_t zero_cnt = 0;
    for (int rd_idx = arr.size() - 1; rd_idx >= 0; rd_idx--) {
        //std::cout << "xx" << arr[rd_idx];
        if (arr[rd_idx] != 0) {
            arr[wr_idx] = arr[rd_idx];
            wr_idx--;
        } else {
            zero_cnt++;
        }
    }
    for (int i = 0; i < zero_cnt; i++) {
        arr[i] = 0;
    }
}

int main() {
    //int arrx[10] = {1, 2, 3, 4, 5, 6, 0};
    std::vector<int> arr;//(arrx, arrx+5);
    arr.push_back(1);
    arr.push_back(2);
    arr.push_back(0);
    arr.push_back(4);
    arr.push_back(5);
    arr.push_back(0);
    arr.push_back(6);
    arr.push_back(0);


    //{};
    compress(arr);
    for (int i = 0; i < arr.size(); i++) {
        std::cout << arr[i] << ",";
    }
    std::cout << std::endl;
}