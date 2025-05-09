from image_pipeline import ImageProcessingPipeline

if __name__ == "__main__":
    folder_path = "c:\\Users\\varda\\Documents\\skittles"
    pipeline = ImageProcessingPipeline(folder_path)
    pipeline.run()
