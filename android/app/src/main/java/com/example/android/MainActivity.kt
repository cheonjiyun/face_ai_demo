package com.example.android

import android.content.Intent
import android.os.Bundle
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import com.example.android.databinding.ActivityMainBinding
import okhttp3.MediaType.Companion.toMediaTypeOrNull
import okhttp3.MultipartBody
import okhttp3.RequestBody.Companion.asRequestBody
import retrofit2.Call
import retrofit2.Response
import java.io.File


class MainActivity : AppCompatActivity() {
    private lateinit var binding: ActivityMainBinding
    private val PICK_IMAGE = 100

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        binding.btnSelectImage.setOnClickListener {
            val intent = Intent(Intent.ACTION_PICK)
            intent.type = "image/*"
            startActivityForResult(intent, PICK_IMAGE)
        }
    }

    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)
        if (requestCode == PICK_IMAGE && resultCode == RESULT_OK) {
            val uri = data?.data
            uri?.let {
                val file = FileUtil.from(this, it)
                uploadImage(file)
            }
        }
    }

    private fun uploadImage(file: File) {
        val requestFile = file.asRequestBody("image/*".toMediaTypeOrNull())
        val body = MultipartBody.Part.createFormData("image", file.name, requestFile)

        ApiClient.api.analyzeFace(body).enqueue(object : retrofit2.Callback<AnalyzeResponse> {
            override fun onResponse(
                call: Call<AnalyzeResponse>,
                response: Response<AnalyzeResponse>
            ) {
                if (response.isSuccessful) {
                    response.body()?.let { result ->
                        showResult(result)
                    }
                } else {
                    Toast.makeText(
                        this@MainActivity,
                        "분석 실패: ${response.code()}",
                        Toast.LENGTH_SHORT
                    ).show()
                }
            }

            override fun onFailure(call: Call<AnalyzeResponse>, t: Throwable) {
                Toast.makeText(this@MainActivity, "에러: ${t.message}", Toast.LENGTH_SHORT).show()
            }
        })

    }

    private fun showResult(result: AnalyzeResponse) {
        val sb = StringBuilder()
        result.proba.forEach {
            val label = it[0] as String
            val prob = (it[1] as Double) * 100
            sb.append("- $label: ${"%.1f".format(prob)}%\n")
        }
        sb.append("\n")
        sb.append(result.gpt)

        binding.tvResult.text = sb.toString()
    }
}
