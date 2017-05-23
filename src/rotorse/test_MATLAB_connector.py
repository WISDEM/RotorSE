import matlab.engine

eng = matlab.engine.start_matlab()

eng.addpath('/Users/bryceingersoll/Dropbox',nargout=0)


test_var = eng.test_connector(2.0)
print (test_var)