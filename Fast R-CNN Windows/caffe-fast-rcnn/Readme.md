# Install CPU-only Caffe with Python Support on Windows
1. clone or download [caffe-windows](https://github.com/BVLC/caffe/tree/windows) <br>
2. copy `.\windows\CommonSettings.props.example` to `.\windows\CommonSettings.props`
3. accordingly change settings in the file CommonSettings.props, including:<br>
	1) set the value of `CpuOnlyBuild` to true<br>
	2) set the value of `UseCuDNN` to false<br>
	3) set the value of `PyhonSupport` to true<br> 
	4) change the value of `PythonDir` to computer's python path<br>
4. open `.\windows\Caffe.sln` with VS2013, and change the configure of the solution to `release` <br>
5. set `Treat Warnings As Errors` to No, which is in:

	PROJECT -> Properties -> Configuration Properties -> C/C++ -> General -> Treat Warnings As Errors<br>  

6. attach the file named `ROIpooling.cpp` to `libcaffe`<br> 
7. build solution, and when it's done, generated files are in `.\Build\x64\Release` <br>