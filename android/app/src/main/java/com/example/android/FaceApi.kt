package com.example.android

import okhttp3.MultipartBody
import retrofit2.Call
import retrofit2.http.Multipart
import retrofit2.http.POST
import retrofit2.http.Part

interface FaceApi {
    @Multipart
    @POST("/api/face-analyze")
    fun analyzeFace(
        @Part image: MultipartBody.Part
    ): Call<AnalyzeResponse>
}
