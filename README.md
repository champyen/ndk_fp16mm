# ndk_fp16mm
NDK r18 ARMv8.2+fp16 matrix multiplication example


build:
$ ndk-build NDK_PROJECT_PATH=. NDK_APPLICATION_MK=Application.mk APP_BUILD_SCRIPT=Android.mk
$ adb push libs/arm64-v8a/fp16mm /data/local/tmp; adb shell "chmod 755 /data/local/tmp/fp16mm"; adb shell "/data/local/tmp/fp16mm"

result under OnePlus 6:
```
[100%] /data/local/tmp/fp16mm
FP16 NEON: 3325 us
NAIVE: 61336 us
DONE!
```
