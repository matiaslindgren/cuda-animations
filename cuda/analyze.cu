#include "device_analysis.hpp"

int main() {
    const DeviceAnalyzer analyzer(0);
    std::cout << "GPU stats:" << std::endl;
    analyzer.print_stats();
}
