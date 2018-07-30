LOCAL_PATH := $(call my-dir)
include $(CLEAR_VARS)
LOCAL_MODULE := fp16mm
LOCAL_CFLAGS += -march=armv8.2a+fp16

LOCAL_SRC_FILES := fp16mm.c

include $(BUILD_EXECUTABLE)
