#include <stdio.h>
#include <stdlib.h> 
#include <time.h>
#include <omp.h>

#define DEBUG 0
#define MAX_RANDOM_INT_SIZE 10


void print_matrix(int** matrix, int rows, int columns) {
#if DEBUG
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < columns; j++) {
            printf("%d ", matrix[i][j]);
        }
        printf("\n");
    }
    printf("\n");
#endif
}

int** generate_matrix(int rows, int columns, int empty = false) {
    int** matrix = (int**)malloc(rows * sizeof(int*));
    if (matrix == NULL) {
        printf("Memory allocation failed.");
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < rows; i++) {
        matrix[i] = (int*)malloc(columns * sizeof(int));
        if (matrix[i] == NULL) {
            printf("Memory allocation failed.");
            exit(EXIT_FAILURE);
        }
        if (!empty) {
            for (int j = 0; j < columns; j++) {
                matrix[i][j] = rand() % MAX_RANDOM_INT_SIZE;
            }
        }
    }
    print_matrix(matrix, rows, columns);

    return matrix;
}


int** sum_matrix(int** matrix1, int** matrix2, int rows, int columns) {
    int** result = generate_matrix(rows, columns, true);

    #pragma omp parallel for shared(matrix1, matrix2, result) reduction(+:result[:rows][:columns])
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < columns; j++) {
            result[i][j] = matrix1[i][j] + matrix2[i][j];
        }
    }

    return result;
}

void cleanup_memory(int** matrix1, int** matrix2, int** sum, int** sum_linear, int rows) {
    for (int i = 0; i < rows; i++) {
        free(matrix1[i]);
        free(matrix2[i]);
        free(sum[i]);
        free(sum_linear[i]);
    }
    free(matrix1);
    free(matrix2);
    free(sum);
    free(sum_linear);
}

int** sum_matrices_linear(int** matrix1, int** matrix2, int rows, int columns) {
    int** result = (int**)malloc(rows * sizeof(int*));
    if (result == NULL) {
        printf("Memory allocation failed.\n");
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < rows; i++) {
        result[i] = (int*)malloc(columns * sizeof(int));
        if (result[i] == NULL) {
            printf("Memory allocation failed.\n");
            exit(EXIT_FAILURE);
        }
        for (int j = 0; j < columns; j++) {
            result[i][j] = matrix1[i][j] + matrix2[i][j];
        }
    }

    return result;
}

int validate_result(int** sum, int** sum_linear, int columns, int rows) {
    int is_valid = true;
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < columns; j++) {
            if (sum[i][j] != sum_linear[i][j]) {
                is_valid = false;
                break;
            }
        }
        if (!is_valid) {
            break;
        }
    }
    return is_valid;
}

int main(int argc, char* argv[]) {
    const int rows = 10000;
    const int columns = 10000;
    srand(time(NULL));
    const int threads_num_max = atoi(argv[1]);

    for (int threads_num = 1; threads_num <= threads_num_max; threads_num++) {
        omp_set_num_threads(threads_num);

        int** matrix1 = generate_matrix(rows, columns);
        int** matrix2 = generate_matrix(rows, columns);

        double start_time = omp_get_wtime();
        int** sum = sum_matrix(matrix1, matrix2, rows, columns);
        double end_time = omp_get_wtime();

        double start_time_linear = omp_get_wtime();
        int** sum_linear = sum_matrices_linear(matrix1, matrix2, rows, columns);
        double end_time_linear = omp_get_wtime();

        int is_valid = validate_result(sum, sum_linear, columns, rows);
        is_valid
            ? printf("Threads %d | Time OMP: %f seconds | Time linear: %f seconds\n", threads_num, end_time - start_time, end_time_linear - start_time_linear)
            : printf("Validation failed - algorithm returned different results than example\n");
        print_matrix(sum, rows, columns);

        cleanup_memory(matrix1, matrix2, sum, sum_linear, rows);
    }

    return 0;
}