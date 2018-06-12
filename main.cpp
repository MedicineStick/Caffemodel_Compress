#include "Pruner.h"
#include<string>
void main(){
	std::string xml_path = "D:\\MINE\\c\\compression\\compression\\compress\\sys_test_config.xml";
	Pruner t = Pruner(xml_path);
	t.start();
	/*int i = 7 / 2;
	std::cout << i;*/
	system("pause");

}




