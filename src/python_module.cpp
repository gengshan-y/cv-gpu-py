#define PY_ARRAY_UNIQUE_SYMBOL pbcvt_ARRAY_API

#include <boost/python.hpp>
#include <pyboostcvconverter/pyboostcvconverter.hpp>

cv::gpu::OpticalFlowDual_TVL1_GPU alg_tvl1;


cv::gpu::PyrLKOpticalFlow flow;
cv::gpu::GpuMat im00,im01,p00, p01, status, flow_u, flow_v;
cv::Mat p01r, flow_x, flow_y, statusr;

namespace pbcvt {

    using namespace boost::python;

/**
 * Example function. Basic inner matrix product using explicit matrix conversion.
 * @param left left-hand matrix operand (NdArray required)
 * @param right right-hand matrix operand (NdArray required)
 * @return an NdArray representing the dot-product of the left and right operands
 */
    PyObject *dot(PyObject *left, PyObject *right) {

        cv::Mat leftMat, rightMat;
        leftMat = pbcvt::fromNDArrayToMat(left);
        rightMat = pbcvt::fromNDArrayToMat(right);
        auto c1 = leftMat.cols, r2 = rightMat.rows;
        // Check that the 2-D matrices can be legally multiplied.
        if (c1 != r2) {
            PyErr_SetString(PyExc_TypeError,
                            "Incompatible sizes for matrix multiplication.");
            throw_error_already_set();
        }
        cv::Mat result = leftMat * rightMat;
        PyObject *ret = pbcvt::fromMatToNDArray(result);
        return ret;
    }

//This example uses Mat directly, but we won't need to worry about the conversion
/**
 * Example function. Basic inner matrix product using implicit matrix conversion.
 * @param leftMat left-hand matrix operand
 * @param rightMat right-hand matrix operand
 * @return an NdArray representing the dot-product of the left and right operands
 */
    cv::Mat dot2(cv::Mat leftMat, cv::Mat rightMat) {
        auto c1 = leftMat.cols, r2 = rightMat.rows;
        if (c1 != r2) {
            PyErr_SetString(PyExc_TypeError,
                            "Incompatible sizes for matrix multiplication.");
            throw_error_already_set();
        }
        cv::Mat result = leftMat * rightMat;

        return result;
    }


/**
 *
 */
    boost::python::tuple optflow_tvl1_gpu(cv::Mat im0, cv::Mat im1) {

        im00.upload(im0);
        im01.upload(im1);
        alg_tvl1(im00,im01,flow_u,flow_v);
        flow_u.download(flow_x);
        flow_v.download(flow_y);
        return boost::python::make_tuple(flow_x, flow_y);
    }


/**
 *
 */
    boost::python::tuple pyrlk_optflow_spr(cv::Mat im0, cv::Mat im1,cv::Mat p0,
                                           cv::Mat p1) {
        im00.upload(im0);
        im01.upload(im1);
        p00.upload(p0);
        p01.upload(p1);
        
        flow.sparse(im00, im01, p00, p01, status, 0);

        status.download(statusr);
        p01.download(p01r);
        return boost::python::make_tuple(statusr, p01r);
    }

    void set_dev(int dev_id) {
      cv::gpu::setDevice(dev_id);
      flow.winSize = cv::Size(5,5);
      flow.maxLevel = 1;
      flow.iters = 30;
      flow.useInitialFlow = true;

      //alg_tvl1.nscales = 1;
      //alg_tvl1.warps = 1;
    }

    void rel_mem() {
      flow.releaseMemory();
      im00.release();
      im01.release();
      p00.release();
      p01.release();
      status.release();
    }


#if (PY_VERSION_HEX >= 0x03000000)

    static void *init_ar() {
#else
        static void init_ar(){
#endif
        Py_Initialize();

        import_array();
        return NUMPY_IMPORT_ARRAY_RETVAL;
    }

    BOOST_PYTHON_MODULE (pbcvt) {
        //using namespace XM;
        init_ar();

        //initialize converters
        to_python_converter<cv::Mat,
                pbcvt::matToNDArrayBoostConverter>();
        pbcvt::matFromNDArrayBoostConverter();

        //expose module-level functions
        def("dot", dot);
        def("dot2", dot2);
        def("optflow_tvl1_gpu", optflow_tvl1_gpu);
        def("pyrlk_optflow_spr", pyrlk_optflow_spr);
        def("set_dev", set_dev);
        def("rel_mem", rel_mem);

    }

} //end namespace pbcvt
