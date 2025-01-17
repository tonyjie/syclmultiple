/*

Licensed under a Creative Commons Attribution-ShareAlike 4.0
International License.

Code by James Reinders, for class at Cornell in September
2023. Based on Exercise 15 of SYCL Academy Code Exercises.

*/

/*******************************************************************

Check https://tinyurl.com/reinders-4class for lots of
information, only some of it is useful for this class.  :)




Known issues:

Crude addition of "blurred_" to front of file name won't
work if there is a directory in the path, so such runs are
rejected.

If the image is too large - the runtime may segment fault -
this code doesn't check for limits (bad, bad, bad!)

********************************************************************/

#define MYDEBUGS
#define DOUBLETROUBLE

#include <algorithm>
#include <array>
#include <cassert>
#include <iostream>
#include <string>
#include <sycl/sycl.hpp>

#include "image_conv.h"

inline constexpr int filterWidth = 44;
// inline constexpr int filterWidth = 88;
inline constexpr int halo = filterWidth / 2;

int main(int argc, char* argv[]) {
  const char* inFile = argv[1];
  char* outFile;

  if (argc == 2) {
    if (strchr(inFile, '/') || strchr(inFile, '\\')) {
      std::cerr << "Sorry, filename cannot include a path.\n";
      exit(1);
    }
    const char* prefix = "blurred_";
    size_t len1 = strlen(inFile);
    size_t len2 = strlen(prefix);
    outFile = (char*)malloc((len1 + len2 + 1) * sizeof(char));
    strcpy(outFile, prefix);
    strcpy(outFile + 8, inFile);
#ifdef MYDEBUGS
    std::cout << "Input file: " << inFile << "\nOutput file: " << outFile
              << "\n";
#endif
  } else {
    std::cerr << "Usage: " << argv[0] << " imagefile\n";
    exit(1);
  }

  auto inImage = util::read_image(inFile, halo);

  auto outImage = util::allocate_image(inImage.width(), inImage.height(),
                                       inImage.channels());

  // The image convolution support code provides a
  // `filter_type` enum which allows us to choose between
  // `identity` and `blur`. The utility for generating the
  // filter data; `generate_filter` takes a `filter_type`
  // and a width.

  auto filter = util::generate_filter(util::filter_type::blur, filterWidth,
                                      inImage.channels());


  //
  // This code tries to grab up to 100 (MAXDEVICES) GPUs.
  // If there are no GPUs, it will get a default device.
  //
#define MAXDEVICES 100

  sycl::queue myQueues[MAXDEVICES];
  int howmany_devices = 0;
  try {
    auto P = sycl::platform(sycl::gpu_selector_v);
    auto RootDevices = P.get_devices();
    // auto C = sycl::context(RootDevices);
    for (auto &D : RootDevices) {
      myQueues[howmany_devices++] = sycl::queue(D,sycl::property::queue::enable_profiling{});
      if (howmany_devices >= MAXDEVICES)
	break;
    }
  } 
  catch (sycl::exception e) {
    howmany_devices = 1;
    myQueues[0] = sycl::queue(sycl::property::queue::enable_profiling{});
  }

#ifdef DEBUGDUMP
  for (int i = 0; i < howmany_devices; ++i) {
    std::cout << "Device: "
	      << myQueues[i].get_device().get_info<sycl::info::device::name>()
	      << " MaxComputeUnits: " << myQueues[i].get_device().get_info<sycl::info::device::max_compute_units>();
    if (myQueues[i].get_device().has(sycl::aspect::ext_intel_device_info_uuid)) {
      auto UUID = myQueues[i].get_device().get_info<sycl::ext::intel::info::device::uuid>();
      char foo[1024];
      sprintf(foo,"\nUUID = %u.%u.%u.%u.%u.%u.%u.%u.%u.%u.%u.%u.%u.%u.%u.%u",
	      UUID[0],UUID[1],UUID[2],UUID[3],UUID[4],UUID[5],UUID[6],UUID[7],
	      UUID[8],UUID[9],UUID[10],UUID[11],UUID[12],UUID[13],UUID[14],UUID[15]);
      std::cout << foo;
    }
    std::cout << "\n";
  }
#endif

  try {
    sycl::queue myQueue1 = myQueues[0];

   
#ifdef MYDEBUGS
    std::cout << "Running on "
              << myQueue1.get_device().get_info<sycl::info::device::name>();
#ifdef SYCL_EXT_INTEL_DEVICE_INFO
#if SYCL_EXT_INTEL_DEVICE_INFO >= 2
    if (myQueue1.get_device().has(sycl::aspect::ext_intel_device_info_uuid)) {
      auto UUID = myQueue1.get_device().get_info<sycl::ext::intel::info::device::uuid>();
      char foo[1024];
      sprintf(foo,"\nUUID = %u.%u.%u.%u.%u.%u.%u.%u.%u.%u.%u.%u.%u.%u.%u.%u",UUID[0],UUID[1],UUID[2],UUID[3],UUID[4],UUID[5],UUID[6],UUID[7],UUID[8],UUID[9],UUID[10],UUID[11],UUID[12],UUID[13],UUID[14],UUID[15]);
      std::cout << foo;
    }
#endif
#endif
    std::cout << "\n";
#endif

#ifdef DOUBLETROUBLE
    sycl::queue myQueue2 = myQueues[ (howmany_devices > 1) ? 1 : 0 ];
    std::cout << "Second queue is running on "
              << myQueue2.get_device().get_info<sycl::info::device::name>();
#ifdef SYCL_EXT_INTEL_DEVICE_INFO
#if SYCL_EXT_INTEL_DEVICE_INFO >= 2
    if (myQueue2.get_device().has(sycl::aspect::ext_intel_device_info_uuid)) {
      auto UUID = myQueue2.get_device().get_info<sycl::ext::intel::info::device::uuid>();
      char foo[1024];
      sprintf(foo,"\nUUID = %u.%u.%u.%u.%u.%u.%u.%u.%u.%u.%u.%u.%u.%u.%u.%u",UUID[0],UUID[1],UUID[2],UUID[3],UUID[4],UUID[5],UUID[6],UUID[7],UUID[8],UUID[9],UUID[10],UUID[11],UUID[12],UUID[13],UUID[14],UUID[15]);
      std::cout << foo;
    }
#endif
#endif
    std::cout << "\n";
#endif

#ifdef DOUBLETROUBLE
    sycl::queue myQueue3 = myQueues[ (howmany_devices > 2) ? 2 : 0 ];
    std::cout << "Third queue is running on "
              << myQueue3.get_device().get_info<sycl::info::device::name>();
#ifdef SYCL_EXT_INTEL_DEVICE_INFO
#if SYCL_EXT_INTEL_DEVICE_INFO >= 2
    if (myQueue3.get_device().has(sycl::aspect::ext_intel_device_info_uuid)) {
      auto UUID = myQueue3.get_device().get_info<sycl::ext::intel::info::device::uuid>();
      char foo[1024];
      sprintf(foo,"\nUUID = %u.%u.%u.%u.%u.%u.%u.%u.%u.%u.%u.%u.%u.%u.%u.%u",UUID[0],UUID[1],UUID[2],UUID[3],UUID[4],UUID[5],UUID[6],UUID[7],UUID[8],UUID[9],UUID[10],UUID[11],UUID[12],UUID[13],UUID[14],UUID[15]);
      std::cout << foo;
    }
#endif
#endif
    std::cout << "\n";
#endif

#ifdef DOUBLETROUBLE
    sycl::queue myQueue4 = myQueues[ (howmany_devices > 3) ? 3 : 0 ];
    std::cout << "Fourth queue is running on "
              << myQueue4.get_device().get_info<sycl::info::device::name>();
#ifdef SYCL_EXT_INTEL_DEVICE_INFO
#if SYCL_EXT_INTEL_DEVICE_INFO >= 2
    if (myQueue4.get_device().has(sycl::aspect::ext_intel_device_info_uuid)) {
      auto UUID = myQueue4.get_device().get_info<sycl::ext::intel::info::device::uuid>();
      char foo[1024];
      sprintf(foo,"\nUUID = %u.%u.%u.%u.%u.%u.%u.%u.%u.%u.%u.%u.%u.%u.%u.%u",UUID[0],UUID[1],UUID[2],UUID[3],UUID[4],UUID[5],UUID[6],UUID[7],UUID[8],UUID[9],UUID[10],UUID[11],UUID[12],UUID[13],UUID[14],UUID[15]);
      std::cout << foo;
    }
#endif
#endif
    std::cout << "\n";
#endif


// ========== Q1 setting prepare ==========
  auto inImgWidth = inImage.width();

  /* Using Two Devices to run the Task*/
  auto inImgHeight_tot = inImage.height();
  auto inImgRemaining = inImgHeight_tot;

  auto inImgHeight_a = inImgRemaining / 9 * 4; // split the image in half
  inImgHeight_a = inImgHeight_a / 32 * 32;  // make it a multiple of 32
  inImgRemaining -= inImgHeight_a;

  auto inImgHeight_b = inImgRemaining  / 5 * 3;
  inImgHeight_b = inImgHeight_b / 32 * 32;
  inImgRemaining -= inImgHeight_b;

  auto inImgHeight_c = inImgRemaining;
  inImgHeight_c = inImgHeight_c / 32 * 32;
  inImgRemaining -= inImgHeight_c;

//  auto inImgHeight_d = inImgRemaining;

  auto channels = inImage.channels();
  auto filterWidth = filter.width();
  auto halo = filter.half_width();

  auto globalRange_a = sycl::range(inImgWidth, inImgHeight_a);
  auto globalRange_b = sycl::range(inImgWidth, inImgHeight_b);
  auto globalRange_c = sycl::range(inImgWidth, inImgHeight_c);
//  auto globalRange_d = sycl::range(inImgWidth, inImgHeight_d);
  auto localRange = sycl::range(1, 8);    // It seems Intel have 8 "threads" per "warp"

  auto ndRange_a = sycl::nd_range(globalRange_a, localRange);
  auto ndRange_b = sycl::nd_range(globalRange_b, localRange);
  auto ndRange_c = sycl::nd_range(globalRange_c, localRange);
//  auto ndRange_d = sycl::nd_range(globalRange_d, localRange);

  auto inBufRange_a =
      sycl::range(inImgHeight_a + (halo * 2), inImgWidth + (halo * 2)) *
      sycl::range(1, channels);
  auto inBufRange_b =
      sycl::range(inImgHeight_b + (halo * 2), inImgWidth + (halo * 2)) *
      sycl::range(1, channels);
  auto inBufRange_c =
      sycl::range(inImgHeight_c + (halo * 2), inImgWidth + (halo * 2)) *
      sycl::range(1, channels);
/* auto inBufRange_d =
      sycl::range(inImgHeight_d + (halo * 2), inImgWidth + (halo * 2)) *
      sycl::range(1, channels); */

  auto outBufRange_a =
      sycl::range(inImgHeight_a, inImgWidth) * sycl::range(1, channels);
  auto outBufRange_b =
      sycl::range(inImgHeight_b, inImgWidth) * sycl::range(1, channels);
  auto outBufRange_c =
      sycl::range(inImgHeight_c, inImgWidth) * sycl::range(1, channels);
/*  auto outBufRange_d =
      sycl::range(inImgHeight_d, inImgWidth) * sycl::range(1, channels); */

  auto filterRange = filterWidth * sycl::range(1, channels);


#ifdef MYDEBUGS
  std::cout << "inImgWidth: " << inImgWidth << "\ninImgHeight: " << inImgHeight_tot
            << "\ninImgHeight_a: " << inImgHeight_a << "\ninImgHeight_b: " << inImgHeight_b
            << "\ninImgHeight_c: " << inImgHeight_c 
            << "\nchannels: " << channels << "\nfilterWidth: " << filterWidth
            << "\nhalo: " << halo << "\n";
#endif


  // Always good to limit scope of accessors,
  // so a good SYCL program will introduce a scope before
  // defining buffers.
  // Remember: While a buffer exists, the data it points
  // to should ONLY be accessed with an accessor. That
  // goes for the host just as much as the device.



  {
    // ======== Picture blurring submit begin ==========
    auto inBuf_a = sycl::buffer{inImage.data(), inBufRange_a};
    auto outBuf_a = sycl::buffer<float, 2>{outBufRange_a};
    outBuf_a.set_final_data(outImage.data());

    auto inBuf_b = sycl::buffer{inImage.data() + inImgHeight_a * channels * (inImgWidth + halo * 2), inBufRange_b};
    auto outBuf_b = sycl::buffer<float, 2>{outBufRange_b};
    outBuf_b.set_final_data(outImage.data() + inImgHeight_a * channels * inImgWidth);

    auto inBuf_c = sycl::buffer{inImage.data() + (inImgHeight_a + inImgHeight_b) * channels * (inImgWidth + halo * 2),
                     inBufRange_c};
    auto outBuf_c = sycl::buffer<float, 2>{outBufRange_c};
    outBuf_c.set_final_data(outImage.data() + (inImgHeight_a + inImgHeight_b) * channels * inImgWidth);

    auto filterBuf = sycl::buffer{filter.data(), filterRange};

#ifdef MYDEBUGS
    auto t1_start = std::chrono::steady_clock::now();  // Start timing
#endif

    sycl::event e1 = myQueue1.submit([&](sycl::handler& cgh1) {
      sycl::accessor inAccessor{inBuf_a, cgh1, sycl::read_only};
      sycl::accessor outAccessor{outBuf_a, cgh1, sycl::write_only};
      sycl::accessor filterAccessor{filterBuf, cgh1, sycl::read_only};

      cgh1.parallel_for(ndRange_a, [=](sycl::nd_item<2> item) {
        auto globalId = item.get_global_id();
        globalId = sycl::id{globalId[1], globalId[0]};

        auto channelsStride = sycl::range(1, channels);
        auto haloOffset = sycl::id(halo, halo);
        auto src = (globalId + haloOffset) * channelsStride;
        auto dest = globalId * channelsStride;

        // 100 is a hack - so the dim is not dynamic
        float sum[/* channels */ 100];
        assert(channels < 100);

        for (size_t i = 0; i < channels; ++i) {
          sum[i] = 0.0f;
        }

        for (int r = 0; r < filterWidth; ++r) {
          for (int c = 0; c < filterWidth; ++c) {
            auto srcOffset =
                sycl::id(src[0] + (r - halo), src[1] + ((c - halo) * channels));
            auto filterOffset = sycl::id(r, c * channels);

            for (int i = 0; i < channels; ++i) {
              auto channelOffset = sycl::id(0, i);
              sum[i] += inAccessor[srcOffset + channelOffset] *
                        filterAccessor[filterOffset + channelOffset];
            }
          }
        }

        for (size_t i = 0; i < channels; ++i) {
          outAccessor[dest + sycl::id{0, i}] = sum[i];
        }
      });
    });


    sycl::event e3 = myQueue3.submit([&](sycl::handler& cgh1) {
      sycl::accessor inAccessor{inBuf_b, cgh1, sycl::read_only};
      sycl::accessor outAccessor{outBuf_b, cgh1, sycl::write_only};
      sycl::accessor filterAccessor{filterBuf, cgh1, sycl::read_only};

      cgh1.parallel_for(ndRange_b, [=](sycl::nd_item<2> item) {
        auto globalId = item.get_global_id();
        globalId = sycl::id{globalId[1], globalId[0]};

        auto channelsStride = sycl::range(1, channels);
        auto haloOffset = sycl::id(halo, halo);
        auto src = (globalId + haloOffset) * channelsStride;
        auto dest = globalId * channelsStride;

        // 100 is a hack - so the dim is not dynamic
        float sum[100];
        assert(channels < 100);

        for (size_t i = 0; i < channels; ++i) {
          sum[i] = 0.0f;
        }

        for (int r = 0; r < filterWidth; ++r) {
          for (int c = 0; c < filterWidth; ++c) {
            auto srcOffset =
                sycl::id(src[0] + (r - halo), src[1] + ((c - halo) * channels));
            auto filterOffset = sycl::id(r, c * channels);

            for (int i = 0; i < channels; ++i) {
              auto channelOffset = sycl::id(0, i);
              sum[i] += inAccessor[srcOffset + channelOffset] *
                        filterAccessor[filterOffset + channelOffset];
            }
          }
        }

        for (size_t i = 0; i < channels; ++i) {
          outAccessor[dest + sycl::id{0, i}] = sum[i];
        }
      });
    });

    sycl::event e4 = myQueue4.submit([&](sycl::handler& cgh1) {
      sycl::accessor inAccessor{inBuf_c, cgh1, sycl::read_only};
      sycl::accessor outAccessor{outBuf_c, cgh1, sycl::write_only};
      sycl::accessor filterAccessor{filterBuf, cgh1, sycl::read_only};

      cgh1.parallel_for(ndRange_c, [=](sycl::nd_item<2> item) {
        auto globalId = item.get_global_id();
        globalId = sycl::id{globalId[1], globalId[0]};

        auto channelsStride = sycl::range(1, channels);
        auto haloOffset = sycl::id(halo, halo);
        auto src = (globalId + haloOffset) * channelsStride;
        auto dest = globalId * channelsStride;

        // 100 is a hack - so the dim is not dynamic
        float sum[100];
        assert(channels < 100);

        for (size_t i = 0; i < channels; ++i) {
          sum[i] = 0.0f;
        }

        for (int r = 0; r < filterWidth; ++r) {
          for (int c = 0; c < filterWidth; ++c) {
            auto srcOffset =
                sycl::id(src[0] + (r - halo), src[1] + ((c - halo) * channels));
            auto filterOffset = sycl::id(r, c * channels);

            for (int i = 0; i < channels; ++i) {
              auto channelOffset = sycl::id(0, i);
              sum[i] += inAccessor[srcOffset + channelOffset] *
                        filterAccessor[filterOffset + channelOffset];
            }
          }
        }

        for (size_t i = 0; i < channels; ++i) {
          outAccessor[dest + sycl::id{0, i}] = sum[i];
        }
      });
    });



    // ======== Q1 submit end ==========


#ifdef MYDEBUGS
    auto t2_start = std::chrono::steady_clock::now();  // Start timing
#endif

// ======== Q2 submit begin ========== 
#ifdef DOUBLETROUBLE
    std::array<int, 200> d4;
    // inspired and based upon:
    // https://cs.uwaterloo.ca/~alopez-o/math-faq/mathtext/node12.html
    // and
    // https://crypto.stanford.edu/pbc/notes/pi/code.html
    // (retrieved September 13, 2023)
    //
    sycl::buffer outD4(d4);
    sycl::event e2 = myQueue2.submit([&](sycl::handler& cgh2) {
      auto outAccessor = outD4.get_access<sycl::access::mode::write>(cgh2);
      cgh2.single_task([=]() {
        int r[2800 + 1];
        int i, k;
        int b, d;
        int c = 0;
	int hold = 0;

        for (i = 0; i < 2800; i++) {
          r[i] = 2000;
        }
        r[2800] = 0;

        for (k = 2800; k > 0; k -= 14) {
          d = 0;

          i = k;
          for (;;) {
            d += r[i] * 10000;
            b = 2 * i - 1;

            r[i] = d % b;
            d /= b;
            i--;
            if (i == 0) break;
            d *= i;
          }
          outAccessor[hold++] = c + d / 10000;
	  c = d % 10000;
        }
      });
    });
#endif
// ======== Q2 submit end ==========

// ======== measurement ==========
#ifdef DOUBLETROUBLE
    // e2.wait(); // make sure all digits are done being computed
    myQueue2.wait();
    // t2 counter
    auto t2_end = std::chrono::steady_clock::now();  // Stop timing
    
    myQueue1.wait();
    // myQueue2.wait();
    myQueue3.wait();
    myQueue4.wait();
    auto t1_end = std::chrono::steady_clock::now();  // Stop timing
    
    sycl::host_accessor myD4(outD4); // the scope of the buffer continues - so we must not use d4[] directly
    std::cout << "First 800 digits of pi: ";
    for (int i = 0; i < 200; ++i) printf("%.4d", myD4[i]);
    std::cout << "\n";
#endif



#ifdef MYDEBUGS
    // Timing code is from our book (2nd edition) -
    // read section on profiling in
    // Chapter 13 that includes figures 13-6 through 13-8.
    // Check https://tinyurl.com/reinders-4class for link
    // to copy of 2nd edition ("Learn SYCL").


    double time2A = (e2.template get_profiling_info<
                         sycl::info::event_profiling::command_end>() -
                     e2.template get_profiling_info<
                         sycl::info::event_profiling::command_start>());

    double time2B = 
        (std::chrono::duration_cast<std::chrono::microseconds>(t2_end - t2_start)
             .count());                     

    // double time2C = 
    //     (std::chrono::duration_cast<std::chrono::microseconds>(t2_end - t1_kernel_launch_end)
    //          .count());

    std::cout << "====== Pi Calculation ( Queue 2 ) =======\n";

    std::cout << "profiling: Operation completed on device2 in " << time2A
              << " nanoseconds (" << time2A / 1.0e9 << " seconds)\n";
    std::cout << "chrono: Operation completed on device2 in " << time2B * 1000
              << " nanoseconds (" << time2B * 1000 / 1.0e9 << " seconds)\n";
    // std::cout << "chrono [end - kernel_launch_end]: Operation completed on device2 in " << time2C
    //           << " nanoseconds (" << time2C / 1.0e9 << " seconds)\n";


    double time1A = (e1.template get_profiling_info<
                         sycl::info::event_profiling::command_end>() -
                     e1.template get_profiling_info<
                         sycl::info::event_profiling::command_start>());
    double time1B = (e3.template get_profiling_info<
                         sycl::info::event_profiling::command_end>() -
                     e3.template get_profiling_info<
                         sycl::info::event_profiling::command_start>());
    double time1C = (e4.template get_profiling_info<
                         sycl::info::event_profiling::command_end>() -
                     e4.template get_profiling_info<
                         sycl::info::event_profiling::command_start>());

    double time1E =
        (std::chrono::duration_cast<std::chrono::microseconds>(t1_end - t1_start)
             .count());
    // double time1C =
    //     (std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1_kernel_begin)
    //          .count());
    // double time1D = 
    //     (std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1_kernel_end)
    //          .count());

    std::cout << "====== Picture Blurring (Queue 1 & 3 & 4) =======\n";
    std::cout << "profiling: Operation completed on device1 in " << time1A
              << " nanoseconds (" << time1A / 1.0e9 << " seconds)\n";
    std::cout << "profiling: Operation completed on device3 in " << time1B
              << " nanoseconds (" << time1B / 1.0e9 << " seconds)\n";
    std::cout << "profiling: Operation completed on device4 in " << time1C
              << " nanoseconds (" << time1C / 1.0e9 << " seconds)\n";
    std::cout << "chrono: Operation completed on device1 , 3, 4 in " << time1E * 1000
              << " nanoseconds (" << time1E * 1000 / 1.0e9 << " seconds)\n";
    // std::cout << "chrono [end - kernel_launch_begin]: Operation completed on device1 in " << time1C
    //           << " nanoseconds (" << time1C / 1.0e9 << " seconds)\n";
    // std::cout << "chrono [end - kernel_launch_end]: Operation completed on device1 in " << time1D
    //           << " nanoseconds (" << time1D / 1.0e9 << " seconds)\n";


    // std::cout << "chrono more than profiling by " << (time1B * 1000 - time1A)
    //           << " nanoseconds (" << (time1B * 1000 - time1A) / 1.0e9
    //           << " seconds)\n";

    double timetot = 
              (std::chrono::duration_cast<std::chrono::microseconds>(t1_end - t1_start)
              .count());
    std::cout << "chrono: Overall Operation completed on device1, 2, 3, 4 in " << timetot * 1000
          << " nanoseconds (" << timetot * 1000 / 1.0e9 << " seconds)\n";


#endif
  }
}
catch (sycl::exception e) {
  std::cout << "Exception caught: " << e.what() << std::endl;
}

util::write_image(outImage, outFile);
}
