import 'dart:convert';
import 'dart:io';
import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'package:image_picker/image_picker.dart';
import 'package:path_provider/path_provider.dart';
import 'package:video_player/video_player.dart';


void main() async {
  WidgetsFlutterBinding.ensureInitialized();
  initPathProvider(); // Ensure path_provider is properly initialized
  runApp(const MyApp());
}

void initPathProvider() async {
  await getApplicationSupportDirectory();
}

class MyApp extends StatelessWidget {
  const MyApp({Key? key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Flutter Demo',
      theme: ThemeData(
        colorScheme: ColorScheme.fromSwatch(primarySwatch: Colors.blue),
        useMaterial3: true,
      ),
      home: const MyHomePage(title: 'Flutter Demo Home Page'),
    );
  }
}

class MyHomePage extends StatefulWidget {
  const MyHomePage({Key? key, required this.title}) : super(key: key);

  final String title;

  @override
  State<MyHomePage> createState() => _MyHomePageState();
}

class _MyHomePageState extends State<MyHomePage> {
  int _counter = 0;

  void _incrementCounter() {
    setState(() {
      _counter++;
    });
  }

  File? _selectedVideo;
  File? _downloadedVideo;

  Future<void> _uploadVideo() async {
    final imagePicker = ImagePicker();
    final XFile? video =
        await imagePicker.pickVideo(source: ImageSource.gallery);
    if (video != null) {
      setState(() {
        _selectedVideo = File(video.path);
      });

      try {
        // Make a GET request to fetch the CSRF token and set the CSRF cookie
        var csrfResponse =
            await http.get(Uri.parse('http://10.5.18.122:5050/csrf_token/'));
        if (csrfResponse.statusCode == 200) {
          // Extract the CSRF token from the response
          var csrfTokenJson = csrfResponse.body;
          var csrfTokenMap = json.decode(csrfTokenJson);
          var csrfToken = csrfTokenMap['csrf_token'];

          // Extract the CSRF cookie from the response headers
          String? csrfCookie = csrfResponse.headers['set-cookie'];

          // Set the CSRF cookie in subsequent requests
          var headers = {'X-CSRFToken': csrfToken};
          if (csrfCookie != null) {
            headers['cookie'] = csrfCookie;
          }

          // Log the headers for debugging
          print('Request Headers: $headers');

          // Create a multipart request
          var request = http.MultipartRequest(
            'POST',
            Uri.parse('http://10.5.18.122:5050/upload/'),
          );

          // Attach headers to the request
          request.headers.addAll(
              headers.map((key, value) => MapEntry(key, value.toString())));

          // Attach the video file to the request
          request.files.add(await http.MultipartFile.fromPath(
            'video',
            _selectedVideo!.path,
          ));

          // Send the request and wait for the response
          var response = await request.send();

          // Check if the request was successful (status code 200)
          if (response.statusCode == 200) {
            print('Video uploaded successfully');

            // Extract the filename from the response
            var videoFileName = await response.stream.bytesToString();

            Map<String, dynamic> jsonResponse = json.decode(videoFileName);

            // Extract the message and filtered video path
            String message = jsonResponse['message'];
            String filteredVideoPath = jsonResponse['filtered_video_path'];

              // Find the last occurrence of '\' in the string
            int lastIndex = filteredVideoPath.lastIndexOf('\\');
  
  // Extract the substring starting from the character after the last '\'
            String fileName = filteredVideoPath.substring(lastIndex + 1);

            print('Message: $message');
            print('Filtered Video Path: $fileName');

            // Download the processed video
            await _downloadProcessedVideo(fileName);
          } else {
            print('Failed to upload video: ${response.reasonPhrase}');
          }
        } else {
          print('Failed to fetch CSRF token: ${csrfResponse.reasonPhrase}');
        }
      } catch (e) {
        print('Error uploading video: $e');
      }
    }
  }

  Future<void> _downloadProcessedVideo(String videoFileName) async {
    try {
      final Directory? appDownloadsDir = await getDownloadsDirectory();
      if (appDownloadsDir == null) {
        print('Downloads directory not available');
        return;
      }

      final savePath = '${appDownloadsDir.path}/$videoFileName';
      final videoFile = File(savePath);

      var response = await http.get(Uri.parse(
          'http://10.5.18.122:5050/get_processed_video/$videoFileName'));

      await videoFile.writeAsBytes(response.bodyBytes);
      print('Video saved to: $savePath');

      setState(() {
        _downloadedVideo = videoFile;
      });
    } catch (e) {
      print('Error downloading processed video: $e');
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        backgroundColor: Theme.of(context).colorScheme.primary,
        title: Text(widget.title),
      ),
      body: Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: <Widget>[
            _selectedVideo != null
                ? Text('Video selected: ${_selectedVideo!.path}')
                : const Text('No video selected'),
            _downloadedVideo != null
                ? Text('Video downloaded: ${_downloadedVideo!.path}')
                : const Text('No video downloaded'),
          ],
        ),
      ),
      floatingActionButton: FloatingActionButton(
        onPressed: _uploadVideo,
        tooltip: 'Upload Video',
        child: const Icon(Icons.upload),
      ),
    );
  }
}
