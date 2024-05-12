import http from "./http-common";

class UploadFilesService {
  upload(file) {
    let formData = new FormData();

    formData.append("file", file);
    console.log(process.env.NEXT_PUBLIC_SERVER_ADDRESS)

    console.log(formData);
    return http.post("upload", formData, {
      headers: {
        "Content-Type": file.type,
      },
      // onUploadProgress,
    });
  }

  getFiles() {
    return http.get("/files");
  }
}

export default new UploadFilesService();