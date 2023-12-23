#include <cstdlib>
#include <iostream>
template<typename T>
class Fifo { 
    private:
        T* data;
        size_t currentCap;
        size_t maxCap;
        size_t wrIdx;
        size_t rdIdx;

    public:
        Fifo(size_t cap) { 
            maxCap = cap;
            currentCap = 0;
            wrIdx = 0;
            rdIdx = 0;
            data = (T*)malloc(sizeof(T) * maxCap);
            if (data == nullptr) {
                std::cout << "malloc failed";
                exit(-1);
            }
        }

        ~Fifo() { 
            free(data);
        }

        bool Push(T& element) {
            if (currentCap < maxCap) {
                data[wrIdx] = element;
                wrIdx++;
                if (wrIdx >= maxCap) {
                    wrIdx = wrIdx % maxCap;
                }
                currentCap++;
                return true;
            } else {
                return false;
            }
        }

        T Pop() {
            if (currentCap > 0) {
                T result = data[rdIdx];
                rdIdx++;
                if (rdIdx >= maxCap) {
                    rdIdx = rdIdx % maxCap;
                }
                currentCap--;
                return result;
            } else {
                return false;
            }
        }

        size_t Size() {
            return currentCap;
        }

        T Top() {
            return data[rdIdx];
        }
 };

 int main() {
    Fifo<int> ff = Fifo<int>(10);
 }