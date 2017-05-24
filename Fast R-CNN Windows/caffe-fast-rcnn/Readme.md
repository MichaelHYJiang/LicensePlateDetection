# Install CPU-only Caffe with Python Support on Windows
1. Clone or download [caffe-windows](https://github.com/BVLC/caffe/tree/windows) <br>
2. Copy `.\windows\CommonSettings.props.example` to `.\windows\CommonSettings.props`
3. Accordingly change settings in the file CommonSettings.props, including:<br>
	1) set the value of `CpuOnlyBuild` to true<br>
	2) set the value of `UseCuDNN` to false<br>
	3) set the value of `PyhonSupport` to true<br> 
	4) change the value of `PythonDir` to computer's python path<br>
4. Open `.\windows\Caffe.sln` with VS2013, and change the configure of the solution to `release` <br>
5. Set `Treat Warnings As Errors` to No, which is in:

	PROJECT -> Properties -> Configuration Properties -> C/C++ -> General -> Treat Warnings As Errors<br>  

6. Attach the file named `ROIpooling.cpp` to `libcaffe`<br> 
7. Build solution, and when it's done, generated files are in `.\Build\x64\Release` <br>
8. It is recommended to copy the directory `caffe` in `.\Build\x64\Release\pycaffe\` to the following location in computer's python home directory: `.\Lib\site-packages\`. Otherwise, caffe home directory needs to be added to system path every time in use.