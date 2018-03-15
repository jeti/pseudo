#ifndef CONSTANTS_HEADER
#define CONSTANTS_HEADER

/* All of the constants throughtout the program are declared here. 
Note that in C++, it is cleaner to declare these as 
"static const int". However, C does not regard such declarations as 
compile-time constants, so the code would be much slower if we did that. 
*/
enum { n_x = 6 };
enum { n_u = 4 };
enum { n_c = 11 };
enum { n_w = 6 };

#ifndef __cplusplus 
    // These values correspond to 6,4,11,6 above
    enum { n_xu = 10 }; // n_x + n_u
    enum { n_vars_per_waypoint = 111 }; // n_c * (n_x + n_u) + 1
    enum { n_vars = 666 }; // (n_c * (n_x + n_u) + 1 ) * n_w
#else
    enum { n_xu = n_x + n_u }; 
    enum { n_vars_per_waypoint = n_c * (n_x + n_u) + 1 };  
    enum { n_vars = (n_c * (n_x + n_u) + 1 ) * n_w }; 
#endif

enum { rep = 1 }; 
enum { verbose = 0 };
enum { legacy = 0 };

#endif /* CONSTANTS_HEADER */