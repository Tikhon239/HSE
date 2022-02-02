from typing import List


class Fibonacci:
    def __init__(self):
        self.cache = [0, 1]

    def __call__(self, n: int) -> int:
        if n < len(self.cache):
            return self.cache[n]
        else:
            fib_n = self(n - 1) + self(n - 2)
            self.cache.append(fib_n)

        return self.cache[n]

    def get_first_n_numbers(self, n: int) -> List[int]:
        self(n)
        return self.cache[:n]


if __name__ == "__main__":
    f = Fibonacci()
    n = int(input("N: "))

    print("First n numbers", f.get_first_n_numbers(n))
