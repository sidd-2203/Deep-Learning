package com.DeepLearningApps.tomatodiseaseclassifier;

import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;

import android.Manifest;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.media.ThumbnailUtils;
import android.net.Uri;
import android.os.Bundle;
import android.provider.MediaStore;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;

import com.DeepLearningApps.tomatodiseaseclassifier.ml.Model;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;

public class MainActivity extends AppCompatActivity {

    Button camera,gallery;
    ImageView imageView;
    TextView textView;
    int imageSize=256;
    String class_names[]={"Tomato_Early_blight", "Tomato_Late_blight", "Tomato_healthy"};


    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        camera= (Button) findViewById(R.id.button);
        gallery= (Button) findViewById(R.id.button2);
        imageView=findViewById(R.id.imageView);
        textView=findViewById(R.id.result);

        camera.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                if(checkSelfPermission(Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED){
                    Intent cameraIntent=new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
                    startActivityForResult(cameraIntent,3);
                }
                else{
                    requestPermissions(new String[]{Manifest.permission.CAMERA},100);
                }
            }

        });
        gallery.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Intent cameraIntent=new Intent(Intent.ACTION_PICK,MediaStore.Images.Media.EXTERNAL_CONTENT_URI);
                startActivityForResult(cameraIntent,1);
            }
        });
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, @Nullable Intent data) {
        if(resultCode==RESULT_OK){
            if(requestCode==3){
                // camera picker
                Bitmap image= (Bitmap) data.getExtras().get("data");
                int dimension=Math.min(image.getWidth(),image.getHeight());
                image= ThumbnailUtils.extractThumbnail(image,dimension,dimension);
                imageView.setImageBitmap(image);

                image=Bitmap.createScaledBitmap(image,imageSize,imageSize,false);
                classifyImage(image);
            }
            else {
                Uri dat=data.getData();
                Bitmap image=null;
                try {
                    image=MediaStore.Images.Media.getBitmap(this.getContentResolver(),dat);
                } catch (IOException e) {
                    throw new RuntimeException(e);
                }
                imageView.setImageBitmap(image);
                image=Bitmap.createScaledBitmap(image,imageSize,imageSize,false);
                classifyImage(image);
            }
        }
        super.onActivityResult(requestCode, resultCode, data);
    }

    public int argmax(float[] a) {
        float re = Float.MIN_VALUE;
        int arg = -1;
        for (int i = 0; i < a.length; i++) {
            if (a[i] > re) {
                re = a[i];
                arg = i;
            }
        }
        return arg;
    }
    private void classifyImage(Bitmap image) {
        try{
            Model model=Model.newInstance(getApplicationContext());
            TensorBuffer inputFeature= TensorBuffer.createFixedSize(new int[]{1,imageSize,imageSize,3}, DataType.FLOAT32);
            ByteBuffer byteBuffer=ByteBuffer.allocateDirect(4*imageSize*imageSize*3);
            byteBuffer.order(ByteOrder.nativeOrder());
            int []intValues= new int[imageSize*imageSize];
            image.getPixels(intValues,0,image.getWidth(),0,0,image.getWidth(),image.getHeight());
            int pixel=0;

            for(int i=0;i<imageSize;i++){
                for(int j=0;j<imageSize;j++){
                    int val=intValues[pixel++]; // RGB
                    byteBuffer.putFloat(((val>>16)&0xFF)*(1.f/1)); // since i have rescaling layer in my model that is why Im not rescaling it between 0 to 1
                    byteBuffer.putFloat(((val>>8)&0xFF)*(1.f/1));
                    byteBuffer.putFloat((val&0xFF)*(1.f/1));
                }
            }
            inputFeature.loadBuffer(byteBuffer);

            Model.Outputs outputs=model.process(inputFeature);
            TensorBuffer outputFeature=outputs.getOutputFeature0AsTensorBuffer();
            float []confidence=outputFeature.getFloatArray();
            int maxPos=argmax(confidence);
            String result="Type : "+class_names[maxPos]+"\nConfidence: "+(confidence[maxPos]*100)+"%";
            textView.setText(result);

            model.close();
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }
}