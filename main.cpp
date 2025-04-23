#include <iostream>
#include <iomanip>
#include <fstream>
#include <cmath>

using namespace std;

// Function Prototypes (Translated Names)
double** read_matrix(const char* fileName, int* rows, int* cols);
double* read_vector(const char* fileName, int* size);
double vector_norm(double* vec, int size); // Calculates the Euclidean (L2) norm
double* product_Ax(double** matrixA, double* vecX, int* rows, int* cols); // Calculates Matrix A * vector x
double* subtract_vectors(double* vec1, double* vec2, int* size); // Calculates vec1 - vec2
double dot_product(double* vec1, double* vec2, int size); // Calculates the dot product of vec1 and vec2
double* conjugate_gradient(double* initial_guess_X, double** matrixA, double* vecB, int size); // TODO: Add tolerance 'e' as a parameter

void print_matrix(double** matrix, int* rows, int* cols);
void print_vector(double* vec, int* size);

int main() {
    int* n = new int; // Matrix A is assumed square for Conjugate Gradient, so rows = cols = n
    int* m = new int; // Although read separately, n will be used for both dimensions of A

    // Read matrix A (n x n) and vector b (n x 1) from files
    double** A = read_matrix("data/matrix.txt", n, n); // Using n for both rows and cols
    // Note: Ensure matrix.txt first line contains n n (same value twice)
    double* b = read_vector("data/vector.txt", n);
    // Note: Ensure vector.txt first line contains n

    cout << "Matrix A:" << endl;
    print_matrix(A, n, n);
    cout << "\n\nVector b:" << endl;
    print_vector(b, n);

    // Initialize guess vector x0 (e.g., vector of ones)
    double* x0 = new double[*n];
    for (int i = 0; i < *n; i++) {
        x0[i] = 1.0;
    }

    cout << "\n\nInitial guess x0:" << endl;
    print_vector(x0, n);

    // Solve Ax = b using the Conjugate Gradient method
    double* result = conjugate_gradient(x0, A, b, *n);

    cout << "\n\nSolution vector x:" << endl;
    print_vector(result, n);

    // Clean up dynamically allocated memory
    for (int i = 0; i < *n; i++) {
        delete[] A[i];
    }
    delete[] A; // Also delete the array of pointers
    delete[] b;
    delete[] x0;
    delete[] result; // Clean up the result vector
    delete n;
    delete m; // Although m wasn't really used after reading A, clean it up.

    return 0;
}

double** read_matrix(const char* fileName, int* rows, int* cols) {
    ifstream fin(fileName);
    if (!fin) {
        cerr << "Error opening matrix file: " << fileName << endl;
        exit(1);
    }
    fin >> *rows;
    fin >> *cols;

    double** matrix = new double* [*rows];

    for (int i = 0; i < *rows; i++) {
        matrix[i] = new double[*cols];
        for (int j = 0; j < *cols; j++) {
            fin >> matrix[i][j];
             if (fin.fail()) {
                cerr << "Error reading matrix data from file: " << fileName << endl;
                // Basic cleanup before exiting
                for(int k=0; k<=i; ++k) delete[] matrix[k];
                delete[] matrix;
                delete rows; // Assuming ownership if passed like in main
                delete cols; // Assuming ownership if passed like in main
                fin.close();
                exit(1);
            }
        }
    }
    fin.close();

    return matrix;
}

double* read_vector(const char* fileName, int* size) {
    ifstream fin(fileName);    // Open the file fileName under the name fin
     if (!fin) {
        cerr << "Error opening vector file: " << fileName << endl;
        exit(1);
    }
    fin >> *size;                 // Assign the first value read from fin to *size

    double* vec = new double[*size];    // Create the vector dynamically

    for (int i = 0; i < *size; i++) {    // Fill the vector from the .txt file
        fin >> vec[i];
        if (fin.fail()) {
                cerr << "Error reading vector data from file: " << fileName << endl;
                // Basic cleanup before exiting
                delete[] vec;
                delete size; // Assuming ownership if passed like in main
                fin.close();
                exit(1);
        }
    }
    fin.close();

    return vec;
}

double vector_norm(double* vec, int size) {
    double norm_squared = 0.0;
    for (int i = 0; i < size; i++) {
        norm_squared += vec[i] * vec[i];
    }
    return sqrt(norm_squared);
}

double* product_Ax(double** matrixA, double* vecV, int* rows, int* cols) {
    // Result vector b will have 'rows' elements.
    double* result_b = new double[*rows];

    // Initialize the result vector b to zeros
    for (int i = 0; i < *rows; i++) {
        result_b[i] = 0.0;
    }

    // Perform the matrix-vector multiplication
    // result_b[i] = sum(matrixA[i][j] * vecV[j] for j in 0..cols-1)
    for (int i = 0; i < *rows; i++) {
        for (int j = 0; j < *cols; j++) {
            result_b[i] += matrixA[i][j] * vecV[j];
        }
    }

    return result_b;
}

double* subtract_vectors(double* vec1, double* vec2, int* size) {
    double* result = new double[*size];
    for (int i = 0; i < *size; i++) {
        result[i] = vec1[i] - vec2[i];
    }
    return result;
}

double dot_product(double* vec1, double* vec2, int size) {
    double product = 0.0;
    for (int i = 0; i < size; i++) {
        product += vec1[i] * vec2[i];
    }
    return product;
}


//=============================================================================================
// Conjugate Gradient Method Implementation
// Solves the linear system Ax = b where A is symmetric and positive-definite.
//=============================================================================================
double* conjugate_gradient(double* X_k, double** A, double* b, int n) {
    // Algorithm uses k for the current iteration index

    double* X_next = new double[n]; // x(k+1) - Stores the next iteration's solution estimate

    // Calculate initial residual: r(0) = b - A*x(0)
    // Note: Potential inefficiency - product_Ax might be calculated multiple times here initially.
    double* Ax0 = product_Ax(A, X_k, &n, &n);
    double* r_k = new double[n];      // r(k) - Residual vector for current iteration k
    for(int i = 0; i < n; i++){
        r_k[i] = b[i] - Ax0[i];
    }
    delete[] Ax0; // Clean up temporary Ax0

    double* p_k = new double[n];      // p(k) - Search direction vector for current iteration k
    // Initial search direction: p(0) = r(0)
    for(int i = 0; i < n; i++){
        p_k[i] = r_k[i];
    }

    // Temporary vectors needed in the loop
    double* r_next = new double[n];   // r(k+1) - Residual vector for next iteration k+1
    double* p_next = new double[n];   // p(k+1) - Search direction vector for next iteration k+1
    double* w = nullptr;              // Temporary vector, typically stores A*p_k

    double alpha;                     // Step length
    double beta;                      // Coefficient for updating search direction
    double rk_norm_sq;                // Squared norm of residual r(k)
    double r_next_norm_sq;            // Squared norm of residual r(k+1)

    const double tolerance_sq = 0.0001 * 0.0001; // Compare squared norm to squared tolerance
    const int max_iterations = 10000;
    int iteration = 0;

    rk_norm_sq = dot_product(r_k, r_k, n); // Initial squared norm

    // Check if initial guess is already good enough
    if (sqrt(rk_norm_sq) < 0.0001) {
         cout << "\n\nInitial guess is already within tolerance." << endl;
         // Copy initial guess to result vector
         for(int i=0; i<n; ++i) X_next[i] = X_k[i];

         // Clean up allocated memory before returning
         delete[] r_k;
         delete[] p_k;
         delete[] r_next;
         delete[] p_next;
         // w is not allocated yet
         return X_next;
    }


    do {
        iteration++;

        // 1. Calculate w = A * p(k)
        delete[] w; // Delete previous w if it exists
        w = product_Ax(A, p_k, &n, &n);

        // 2. Calculate step length alpha(k) = (r(k)^T * r(k)) / (p(k)^T * A * p(k))
        //    alpha = dot_product(r_k, r_k, n) / dot_product(p_k, w, n);
        //    Using pre-calculated rk_norm_sq for efficiency.
        alpha = rk_norm_sq / dot_product(p_k, w, n);


        // 3. Update solution estimate: x(k+1) = x(k) + alpha(k) * p(k)
        for (int j = 0; j < n; j++) {
            X_next[j] = X_k[j] + (alpha * p_k[j]);
        }

        // 4. Update residual: r(k+1) = r(k) - alpha(k) * w   (where w = A*p(k))
        for (int j = 0; j < n; j++) {
            r_next[j] = r_k[j] - (alpha * w[j]);
        }

        // 5. Check convergence: ||r(k+1)||^2 < tolerance^2 ?
        r_next_norm_sq = dot_product(r_next, r_next, n);
        if (r_next_norm_sq < tolerance_sq) {
            cout << "\n\nTolerance reached in " << iteration << " iterations.";
            cout << " (Residual norm squared: " << r_next_norm_sq << ")" << endl;

            // Clean up before returning the solution
             delete[] r_k;
             delete[] p_k;
             delete[] r_next;
             delete[] p_next;
             delete[] w;
            return X_next; // Return the calculated solution X_next
        }

        // 6. Calculate beta(k) = (r(k+1)^T * r(k+1)) / (r(k)^T * r(k))
        beta = r_next_norm_sq / rk_norm_sq;

        // 7. Update search direction: p(k+1) = r(k+1) + beta(k) * p(k)
        for (int j = 0; j < n; j++) {
            p_next[j] = r_next[j] + (beta * p_k[j]);
        }

        // --- Prepare for the next iteration (k becomes k+1) ---
        rk_norm_sq = r_next_norm_sq; // Store current norm squared for the next alpha calculation

        for (int i = 0; i < n; i++) {
            X_k[i] = X_next[i]; // Update X_k for the next loop iteration
            r_k[i] = r_next[i]; // Update r_k
            p_k[i] = p_next[i]; // Update p_k
        }

    } while (iteration < max_iterations);

    // Loop finished without meeting tolerance
    if (iteration == max_iterations) {
        cout << "\n\nMaximum iterations (" << max_iterations << ") reached. Solution might not have converged.";
         cout << " (Final residual norm squared: " << rk_norm_sq << ")" << endl;
    }

    // Clean up dynamically allocated memory within the function
    delete[] r_k;
    delete[] p_k;
    delete[] r_next;
    delete[] p_next;
    delete[] w;

    // Return the last calculated solution estimate
    return X_next;
}
//==============================================================================================

void print_matrix(double** matrix, int* rows, int* cols) {
    cout << fixed << setprecision(4); // Format output
    for (int i = 0; i < *rows; i++) {
        for (int j = 0; j < *cols; j++) {
            cout << setw(10) << matrix[i][j] << " ";
        }
         cout << "\n"; // Newline after each row
    }
    cout << defaultfloat << setprecision(6); // Reset formatting
}

void print_vector(double* vec, int* size) {
     cout << fixed << setprecision(4); // Format output
    for (int i = 0; i < *size; i++) {
        cout << setw(10) << vec[i]; // Use setw for alignment similar to matrix
        if (i < *size - 1) {
            cout << "\n"; // Newline between elements for column vector look
        }
    }
    cout << endl; // Final newline
    cout << defaultfloat << setprecision(6); // Reset formatting
}