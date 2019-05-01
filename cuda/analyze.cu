#include "device_analysis.hpp"
#include <cstdlib>

int main(int argc, char** argv) {
	int device_num = 0;
	if (argc == 2) {
		device_num = std::stoi(argv[1]);
	}
	const DeviceAnalyzer analyzer(device_num);
	std::cout << "Device " << device_num << " stats:\n";
	analyzer.print_stats();
}
