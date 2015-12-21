/* 
 * File:   main.cpp
 * Author: Abuenameh
 *
 * Created on 17 November 2014, 22:05
 */

#include <cstdlib>
#include <queue>

using namespace std;

#include <boost/lexical_cast.hpp>
#include <boost/thread.hpp>
#include <boost/progress.hpp>
#include <boost/random.hpp>
#include <boost/filesystem.hpp>
#include <boost/filesystem/fstream.hpp>
#include <boost/date_time.hpp>
#include <boost/interprocess/containers/vector.hpp>
#include <boost/interprocess/managed_shared_memory.hpp>
#include <boost/interprocess/containers/string.hpp>
#include <boost/interprocess/sync/interprocess_condition.hpp>
#include <boost/interprocess/sync/interprocess_mutex.hpp>
#include <boost/interprocess/sync/scoped_lock.hpp>

using namespace boost;
using namespace boost::random;
using namespace boost::filesystem;
using namespace boost::posix_time;
//using namespace boost::interprocess;

using boost::interprocess::shared_memory_object;
using boost::interprocess::managed_shared_memory;
using boost::interprocess::create_only;
using boost::interprocess::open_only;
//using boost::interprocess::allocator;
using boost::interprocess::basic_string;
using boost::interprocess::interprocess_mutex;
using boost::interprocess::interprocess_condition;
//using boost::interprocess::interprocess_;

#include <boost/process.hpp>

using namespace boost::process;
using namespace boost::process::initializers;

#include <nlopt.hpp>

using namespace nlopt;

#include "gutzwiller.hpp"
//#include "mathematica.hpp"
#include "casadi.hpp"
#include "casadimath.hpp"
//#include "orderparameter.hpp"

#include <casadi/interfaces/sundials/cvodes_interface.hpp>

typedef managed_shared_memory::segment_manager segment_manager_t;

typedef interprocess::allocator<void, segment_manager_t> void_allocator;

typedef interprocess::allocator<double, segment_manager_t> double_allocator;
typedef interprocess::vector<double, double_allocator> double_vector;

typedef interprocess::allocator<complex<double>, segment_manager_t> complex_allocator;
typedef interprocess::vector<complex<double>, complex_allocator> complex_vector;
typedef interprocess::allocator<complex_vector, segment_manager_t> complex_vector_allocator;
typedef interprocess::vector<complex_vector, complex_vector_allocator> complex_vector_vector;

typedef interprocess::allocator<char, segment_manager_t> char_allocator;
typedef interprocess::basic_string<char, std::char_traits<char>, char_allocator> char_string;

struct worker_input {
    double Wi;
    double Wf;
    double mu;
    double scale;
    double_vector xi;
    double U0;
    double_vector J0;
    double_vector x0;
    complex_vector_vector f0;
    char_string integrator;
    double dt;

    worker_input(const void_allocator& void_alloc) : xi(void_alloc), J0(void_alloc), x0(void_alloc), f0(void_alloc), integrator(void_alloc) {
    }
};

struct worker_tau {
    interprocess_mutex mutex;
    interprocess_condition cond_empty;
    interprocess_condition cond_full;

    double tau;

    bool full;

    worker_tau() : full(false) {
    }
};

struct worker_output {
    double Ei;
    double Ef;
    double Q;
    double p;
    double_vector Es;
    complex_vector b0;
    complex_vector bf;
    complex_vector_vector f0;
    complex_vector_vector ff;
    char_string runtime;
    bool success;

    interprocess_mutex mutex;
    interprocess_condition cond_empty;
    interprocess_condition cond_full;

    bool full;

    worker_output(const void_allocator& void_alloc) : Es(void_alloc), b0(void_alloc), bf(void_alloc), f0(void_alloc), ff(void_alloc), runtime(void_alloc), full(false) {
    }
};

struct results {
    double tau;
    double Ei;
    double Ef;
    double Q;
    double p;
    double U0;
    vector<double> Es;
    vector<double> J0;
    vector<complex<double>> b0;
    vector<complex<double>> bf;
    vector<vector<complex<double>>> f0;
    vector<vector<complex<double>>> ff;
    string runtime;

//    results(const void_allocator& void_alloc) : Es(void_alloc), J0(void_alloc), b0(void_alloc), bf(void_alloc), f0(void_alloc), ff(void_alloc), runtime(void_alloc) {
//    }
};

double UWi(double W) {
    return -2 * (g24 * g24) / Delta * (Ng * Ng * W * W) / ((Ng * Ng + W * W) * (Ng * Ng + W * W));
}

struct input {
    double tau;
};

boost::mutex progress_mutex;
boost::mutex inputs_mutex;
boost::mutex problem_mutex;

boost::random::mt19937 rng;
boost::random::uniform_real_distribution<> uni(-1, 1);

complex<double> dot(vector<complex<double>>& v, vector<complex<double>>& w) {
    complex<double> res = 0;
    for (int i = 0; i < v.size(); i++) {
        res += ~v[i] * w[i];
    }
    return res;
}


worker_input* initialize(double Wi, double Wf, double mu, double scale, vector<double>& xi, managed_shared_memory& segment) {

    SX f = SX::sym("f", 2 * L * dim);
    SX dU = SX::sym("dU", L);
    SX J = SX::sym("J", L);
    SX U0 = SX::sym("U0");

    U0 = UW(Wi)/scale;
    for (int i = 0; i < L; i++) {
        J[i] = JWij(Wi * xi[i], Wi * xi[mod(i + 1)])/scale;
        dU[i] = UW(Wi * xi[i])/scale - U0;
    }

//    SX E = energy(f, J, U0, dU, mu/scale);
    SX E2 = energyc(f, J, U0, dU, mu/scale, false);
    
    SX g = SX::sym("g", L);
    for (int i = 0; i < L; i++) {
        g[i] = 0;
        for (int n = 0; n < dim; n++) {
            g[i] += f[2*(i*dim+n)]*f[2*(i*dim+n)] + f[2*(i*dim+n)+1]*f[2*(i*dim+n)+1];
        }
    }

//    SXFunction nlp("nlp", nlpIn("x", f), nlpOut("f", E));
//    NlpSolver solver("solver", "ipopt", nlp, make_dict("hessian_approximation", "limited-memory", "linear_solver", "ma86", "print_level", 0, "print_time", false, "obj_scaling_factor", 1));
//    NlpSolver solver("solver", "ipopt", nlp, make_dict("hessian_approximation", "limited-memory", "linear_solver", "ma86", "print_level", 0, "print_time", false));
    SXFunction nlp2("nlp", nlpIn("x", f), nlpOut("f", E2, "g", g));
    NlpSolver solver2("solver", "ipopt", nlp2, make_dict("hessian_approximation", "limited-memory", "linear_solver", "ma86", "print_level", 0, "print_time", false, "obj_scaling_factor", 1));

    boost::random::mt19937 rng;
    boost::random::uniform_real_distribution<> uni(-1, 1);

    vector<double> xrand(2 * L*dim, 1);
    rng.seed();
    for (int i = 0; i < 2 * L * dim; i++) {
        xrand[i] = uni(rng);
//        xrand[i] = 1./sqrt(2*L*dim);
    }

    map<string, DMatrix> arg;
    arg["lbx"] = -1;
    arg["ubx"] = 1;
    arg["x0"] = xrand;
    arg["lbg"] = 1;
    arg["ubg"] = 1;

    map<string, DMatrix> res = solver2(arg);
    vector<double> x0 = res["x"].nonzeros();
//    vector<double> x0 = xrand;
//    cout << "x0 = " << ::math(x0) << endl;
//    cout << "E0 = " << ::math(res["f"].toScalar()) << endl;

    vector<complex<double>> x0i(dim);
    for (int i = 0; i < L; i++) {
        for (int n = 0; n <= nmax; n++) {
            x0i[n] = complex<double>(x0[2 * (i * dim + n)], x0[2 * (i * dim + n) + 1]);
        }
        double nrm = sqrt(abs(dot(x0i, x0i)));
        for (int n = 0; n <= nmax; n++) {
            x0[2 * (i * dim + n)] /= nrm;
            x0[2 * (i * dim + n) + 1] /= nrm;
        }
    }
//    cout << "nlp: " << ::math(nlp(vector<DMatrix>{x0, vector<double>()})[0].toScalar()) << endl;

    void_allocator void_alloc(segment.get_segment_manager());
    worker_input* input = segment.construct<worker_input>("input")(void_alloc);
    input->U0 = UW(Wi);
    for (int i = 0; i < L; i++) {
        input->J0.push_back(JWij(Wi * xi[i], Wi * xi[mod(i + 1)]));
    }
    for (int i = 0; i < 2 * L * dim; i++) {
        input->x0.push_back(x0[i]);
    }
    for (int i = 0; i < L; i++) {
        complex_vector f0i(dim, void_alloc);
        for (int n = 0; n <= nmax; n++) {
            f0i[n] = complex<double>(x0[2 * (i * dim + n)], x0[2 * (i * dim + n) + 1]);
        }
        input->f0.push_back(f0i);
    }
    input->Wi = Wi;
    input->Wf = Wf;
    input->mu = mu;
    input->scale = scale;
    for (int i = 0; i < L; i++) {
        input->xi.push_back(xi[i]);
    }

    return input;
}

complex<double> dot(complex_vector& v, complex_vector& w) {
    complex<double> res = 0;
    for (int i = 0; i < v.size(); i++) {
        res += ~v[i] * w[i];
    }
    return res;
}

/*
 * 
 */
int main(int argc, char** argv) {
    
//    build_odes();
//    return 0;

    ptime begin = microsec_clock::local_time();

    random::mt19937 rng;
    random::uniform_real_distribution<> uni(-1, 1);

    int seed = lexical_cast<int>(argv[1]);
    
    if (seed != -1) {

        double Wi = lexical_cast<double>(argv[2]);
        double Wf = lexical_cast<double>(argv[3]);

        double mu = lexical_cast<double>(argv[4]);
        
        double scale = lexical_cast<double>(argv[5]);

        double Ui = UWi(Wi);

        double D = lexical_cast<double>(argv[6]);

        double taui = lexical_cast<double>(argv[7]);
        double tauf = lexical_cast<double>(argv[8]);
        int ntaus = lexical_cast<int>(argv[9]);

        int numthreads = lexical_cast<int>(argv[10]);

        int resi = lexical_cast<int>(argv[11]);

        //        int integrator = lexical_cast<int>(argv[11]);
        std::string intg = argv[12];
        double dt = lexical_cast<double>(argv[13]);

#ifdef AMAZON
        //    path resdir("/home/ubuntu/Results/Canonical Transformation Dynamical Gutzwiller");
        path resdir("/home/ubuntu/Dropbox/Amazon EC2/Simulation Results/Canonical Transformation Dynamical Gutzwiller 2");
#else
        path resdir("/Users/Abuenameh/Documents/Simulation Results/Canonical Transformation Dynamical Gutzwiller 2");
        //        path resdir("/Users/Abuenameh/Documents/Simulation Results/Dynamical Gutzwiller Hartmann Comparison");
#endif
        if (!exists(resdir)) {
            cerr << "Results directory " << resdir << " does not exist!" << endl;
            exit(1);
        }
        ostringstream oss;
        oss << "res." << resi << ".txt";
        path resfile = resdir / oss.str();
        //#ifndef AMAZON
        while (exists(resfile)) {
            resi++;
            oss.str("");
            oss << "res." << resi << ".txt";
            resfile = resdir / oss.str();
        }
        //#endif
        if (seed < 0) {
            resi = seed;
            oss.str("");
            oss << "res." << resi << ".txt";
            resfile = resdir / oss.str();
        }
        vector<double> xi(L, 1);
        rng.seed(seed);
        if (seed > -1) {
            for (int j = 0; j < L; j++) {
                xi[j] = (1 + D * uni(rng));
            }
        }

        //        double Ui = UWi(Wi);
        double mui = mu * Ui;

        struct shm_remove {

            shm_remove() {
                shared_memory_object::remove("SharedMemory");
            }

            ~shm_remove() {
                shared_memory_object::remove("SharedMemory");
            }
        } remover;

        int size = 1000 * (sizeof (worker_input) + numthreads * (sizeof (worker_tau) + sizeof (worker_output))); //2 * (((2 * L * dim + L + 1) + numthreads * (4 * L * dim + 5 * L + 6)) * sizeof (double) +numthreads * 2 * sizeof (ptime)/*sizeof(time_period)*/);

        managed_shared_memory segment(create_only, "SharedMemory", size);

        worker_input* w_input = initialize(Wi, Wf, mui, scale, xi, segment);
//        return 0;
        
        ostringstream pyoss;
        pyoss << "input_" << L << "_" << seed << "_" << D << ".bin";
        path pypath = resdir / pyoss.str();
        filesystem::ofstream pyfile(pypath);
        pyfile.write(reinterpret_cast<char*>(xi.data()), L*sizeof(double));
        for (int i = 0; i < L; i++) {
            for (int n = 0; n <= nmax; n++) {
                double re = w_input->f0[i][n].real();
                double im = w_input->f0[i][n].imag();
                pyfile.write(reinterpret_cast<char*>(&re), sizeof(double));
                pyfile.write(reinterpret_cast<char*>(&im), sizeof(double));
            }
        }
        pyfile.close();
        exit(0);
    }
    else {
    }

    return 0;

}

