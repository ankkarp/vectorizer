import FileInput from "../components/blocks/FileInput/FileInput";
import ResultBlock from "../components/blocks/ResultBlock/ResultBlock";
import UploadIcon from "../components/icons/UploadIcon/UploadIcon";
import MainLayout from "../components/layouts/MainLayout/MainLayout";

export default function Home() {
  return (
    <MainLayout>
      <FileInput />
      <ResultBlock />
    </MainLayout>
  );
}
