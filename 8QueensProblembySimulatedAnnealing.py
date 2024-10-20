import numpy as np
import random

# Class Node dùng để biểu diễn một trạng thái trong bài toán
class Node:
    def __init__(self, state, parent=None, action=None):
        self.state = state  # Trạng thái của bàn cờ, là một tuple
        self.parent = parent  # Nút cha, để theo dõi hành trình tìm kiếm
        self.action = action  # Hành động để đi từ trạng thái cha đến trạng thái con

    # Phương thức expand để tạo danh sách các trạng thái con dựa trên các hành động có thể thực hiện được
    def expand(self, problem):
        return [self.child_node(problem, action) for action in problem.actions(self.state)]

    # Phương thức child_node tạo ra một nút con dựa trên hành động cụ thể
    def child_node(self, problem, action):
        next_state = problem.result(self.state, action)  # Trạng thái tiếp theo sau khi thực hiện hành động
        return Node(next_state, self, action)  # Tạo nút mới với trạng thái đó

# Class NQueensProblem biểu diễn bài toán 8 con hậu
class NQueensProblem:
    def __init__(self, N):
        self.initial = tuple([-1] * N)  # Khởi tạo trạng thái ban đầu với N cột đều là -1 (chưa đặt quân hậu nào)
        self.N = N  # Số lượng con hậu cần được đặt

    # Phương thức actions trả về các hành động có thể thực hiện (hàng nào có thể đặt quân hậu trong cột hiện tại)
    def actions(self, state):
        if state[-1] != -1:  # Nếu đã đặt đủ tất cả các quân hậu
            return []
        else:
            col = state.index(-1)  # Tìm cột đầu tiên chưa đặt quân hậu
            return [row for row in range(self.N) if not self.conflicted(state, row, col)]  # Trả về danh sách hàng có thể đặt hậu

    # Phương thức result trả về trạng thái mới sau khi đặt một quân hậu vào hàng row trong cột col
    def result(self, state, row):
        col = state.index(-1)  # Tìm cột đầu tiên chưa được đặt quân hậu
        new = list(state[:])  # Tạo bản sao của trạng thái hiện tại
        new[col] = row  # Đặt quân hậu vào hàng 'row' trong cột 'col'
        return tuple(new)  # Trả về trạng thái mới

    # Phương thức conflicted kiểm tra xem việc đặt một quân hậu ở (row, col) có gây xung đột với các quân hậu đã đặt không
    def conflicted(self, state, row, col):
        return any(self.conflict(row, col, state[i], i) for i in range(col))  # Kiểm tra xung đột với các quân hậu đã đặt

    # Phương thức conflict kiểm tra nếu hai quân hậu ở (row1, col1) và (row2, col2) có xung đột không
    def conflict(self, row1, col1, row2, col2):
        return (row1 == row2 or  # Cùng hàng
                col1 == col2 or  # Cùng cột
                row1 - col1 == row2 - col2 or  # Cùng đường chéo chính
                row1 + col1 == row2 + col2)  # Cùng đường chéo phụ

    # Phương thức value tính toán số lượng xung đột trong một trạng thái
    def value(self, node):
        number_conflicts = 0  # Biến đếm số lượng xung đột
        for (r1, c1) in enumerate(node.state):  # Duyệt qua từng cặp quân hậu
            for (r2, c2) in enumerate(node.state):
                if (r1, c1) != (r2, c2):  # Không so sánh với chính nó
                    number_conflicts += self.conflict(r1, c1, r2, c2)  # Nếu có xung đột, tăng biến đếm
        return -number_conflicts  # Trả về số xung đột dưới dạng âm (vì càng ít xung đột thì giá trị càng cao)

    # Phương thức schedule là hàm làm lạnh, tính toán nhiệt độ cho từng thời điểm t
    def schedule(self, t, k=20, lam=0.005, limit=1000):
        return (k * np.exp(-lam * t) if t < limit else 0)

# Hàm simulated_annealing thực hiện giải thuật Simulated Annealing để giải bài toán N quân hậu
def simulated_annealing(problem):
    current = Node(problem.initial)  # Khởi tạo trạng thái ban đầu
    current_cost = problem.value(current)  # Tính toán chi phí của trạng thái ban đầu

    T = 1000  # Nhiệt độ ban đầu
    step = 0  # Biến đếm số bước

    while T > 0.1:  # Duy trì đến khi nhiệt độ nhỏ hơn 0.1
        neighbors = current.expand(problem)  # Lấy các trạng thái lân cận
        if not neighbors:  # Nếu không còn trạng thái lân cận nào, thoát khỏi vòng lặp
            break

        next_node = random.choice(neighbors)  # Chọn ngẫu nhiên một trạng thái lân cận
        next_cost = problem.value(next_node)  # Tính chi phí của trạng thái đó

        delta_cost = next_cost - current_cost  # Tính độ thay đổi chi phí

        # Chấp nhận trạng thái mới nếu chi phí tốt hơn (delta_cost > 0) hoặc ngẫu nhiên theo hàm exp
        if delta_cost > 0 or (random.uniform(0, 1) < np.exp(delta_cost / T)):
            current = next_node
            current_cost = next_cost

        print(f"Step {step}:")  # In ra bước hiện tại
        print_board(current.state, problem.N)  # In trạng thái của bàn cờ
        print(f"Cost: {current_cost}\n")  # In chi phí hiện tại

        step += 1
        T *= 0.99  # Giảm nhiệt độ dần dần

    return current, current_cost  # Trả về trạng thái cuối cùng và chi phí

# Hàm print_board dùng để in trạng thái của bàn cờ
def print_board(state, N):
    board = [['-' for _ in range(N)] for _ in range(N)]  # Khởi tạo bàn cờ N x N với dấu '-'
    for col, row in enumerate(state):
        if row != -1:  # Nếu quân hậu đã được đặt ở hàng 'row', cột 'col'
            board[row][col] = 'Q'  # Đặt quân hậu (ký tự 'Q')
    for row in board:
        print(" ".join(row))  # In mỗi hàng của bàn cờ
    print("\n")  # In dòng trống sau mỗi bàn cờ

# Hàm main để khởi chạy chương trình
if __name__ == '__main__':
    number_of_queen = 8
    problem1 = NQueensProblem(number_of_queen)

    result, cost = simulated_annealing(problem1)
    print("Kết quả cuối cùng:")
    print_board(result.state, number_of_queen)
    print("Cost: ", cost)
#Link video: https://drive.google.com/file/d/1nd0m9CxXWo86OLs6ZZ-YC5MrmLkTxeod/view?usp=sharing
# Code có sự tham khảo từ file thuật toán của cô trong group Zalo
