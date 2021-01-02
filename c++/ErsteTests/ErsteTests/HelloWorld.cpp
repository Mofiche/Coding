#include <iostream>
#include <math.h>

int main() {

	double a, b, c,x1, x2, ex, ey = 0;
	bool max = true;
	std::cout << "Bitte geben sie die Koeffizienten der quadratischen Kurve in Form von ax^2 + bx +c an \n \n";
	std::cout << "a?" << std::endl;
	std::cin >> a;
	std::cout << std::endl;
	std::cout << "b?" << std::endl;
	std::cin >> b;
	std::cout << std::endl;
	std::cout << "c?" << std::endl;
	std::cin >> c;
	std::cout << std::endl;

	x1 = (-b + sqrt( (b*b)- (4*a*c))) / (2 * a);
	x2 = (-b - sqrt((b*b) - (4 * a*c))) / (2 * a);

	std::cout << "Nullstellen : \n x1 : \n " << x1 << "\n x2 : \n " << x2 <<std::endl ;

	ex = - b / (2*a);
	ey = (a * ex*ex) + (b * ex) + c;

	std::cout << "Extremstelle : \n";
	if (a > 0) max = false;
		else max = true;

	if(max) std::cout << "H(" << ex << "|" << ey << ")" << std::endl;
		else std::cout << "T(" << ex << "|" << ey << ")" << std::endl;


	system("pause");
	return 0;
}