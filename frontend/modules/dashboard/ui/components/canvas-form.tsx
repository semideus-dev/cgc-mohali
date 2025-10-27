"use client";

import { toast } from "sonner";
import { useForm } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";
import { z } from "zod";
import { Button } from "@/components/ui/button";
import {
  Form,
  FormControl,
  FormDescription,
  FormField,
  FormItem,
  FormLabel,
  FormMessage,
} from "@/components/ui/form";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { DropzoneOptions } from "react-dropzone";
import {
  FileInput,
  FileUploader,
  FileUploaderContent,
  FileUploaderItem,
} from "@/components/ui/file-upload";
import { cn } from "@/lib/utils";
import { useState } from "react";
import { Upload, Loader2 } from "lucide-react";
import { useUploadThing } from "@/lib/uploadthing";
import { useCreateCanvas } from "@/modules/canvas/server/hooks";

const formSchema = z.object({
  name: z
    .string()
    .min(3, { message: "Name must be at least 3 characters long." })
    .max(50, { message: "Name cannot exceed 50 characters." }),
  prompt: z
    .string()
    .min(10, { message: "Please provide a detailed prompt (min 10 chars)." })
    .max(500, { message: "Prompt cannot exceed 500 characters." }),
  file: z
    .array(
      z.instanceof(File).refine((file) => file.size < 10 * 1024 * 1024, {
        message: "File size must be less than 10MB",
      })
    )
    .max(1, {
      message: "Maximum 1 file is allowed",
    })
    .nullable(),
});

type FormValues = z.infer<typeof formSchema>;

interface CanvasFormProps {
  onSuccess?: () => void;
}

export default function CanvasForm({ onSuccess }: CanvasFormProps) {
  const form = useForm<FormValues>({
    resolver: zodResolver(formSchema),
    defaultValues: {
      name: "",
      prompt: "",
      file: null,
    },
  });

  const [isUploading, setIsUploading] = useState(false);

  // UploadThing hook
  const { startUpload } = useUploadThing("canvasImageUploader", {
    onClientUploadComplete: (res) => {
      console.log("Files uploaded:", res);
    },
    onUploadError: (error: Error) => {
      console.error("Upload error:", error);
      toast.error(`Upload failed: ${error.message}`);
    },
    onUploadBegin: (name) => {
      console.log("Upload started:", name);
    },
  });

  // Canvas creation hook
  const createCanvasMutation = useCreateCanvas();

  const dropzoneConfig = {
    multiple: false,
    maxFiles: 1,
    maxSize: 10 * 1024 * 1024,
    accept: {
      "image/*": [".png", ".jpg", ".jpeg", ".webp"],
    },
  } satisfies DropzoneOptions;

  async function onSubmit(values: FormValues) {
    try {
      setIsUploading(true);

      let imageUrl: string | undefined;

      // Upload file if provided
      if (values.file && values.file.length > 0) {
        toast.info("Uploading image...");

        const uploadResult = await startUpload(values.file);

        if (!uploadResult || uploadResult.length === 0) {
          throw new Error("Failed to upload file");
        }

        imageUrl = uploadResult[0].url;
        toast.success("Image uploaded successfully!");
      }

      // Create canvas in database
      toast.info("Creating canvas...");

      await createCanvasMutation.mutateAsync({
        name: values.name,
        prompt: values.prompt,
        url: imageUrl,
      });

      // Reset form on success
      form.reset();

      // Call success callback if provided
      onSuccess?.();
    } catch (error) {
      console.error("Form submission error", error);

      const errorMessage =
        error instanceof Error
          ? error.message
          : "Failed to create canvas. Please try again.";

      toast.error(errorMessage);
    } finally {
      setIsUploading(false);
    }
  }

  const isSubmitting = isUploading || createCanvasMutation.isPending;

  return (
    <Form {...form}>
      <form
        onSubmit={form.handleSubmit(onSubmit)}
        className="space-y-8 my-4 max-w-lg"
      >
        {/* File Upload */}
        <FormField
          control={form.control}
          name="file"
          render={({ field }) => (
            <FormItem>
              <FormLabel>Canvas Image</FormLabel>
              <FileUploader
                value={field.value}
                onValueChange={field.onChange}
                dropzoneOptions={dropzoneConfig}
                reSelect={true}
              >
                <FileInput className="w-full border p-5 items-center justify-center flex flex-col gap-2 text-muted-foreground border-dashed">
                  <Upload className="h-6 w-6" />
                  <span>Select an image (.png / .jpg / .jpeg / .webp)</span>
                  <span className="text-xs">Optional - Max 4MB</span>
                </FileInput>
                {field.value && field.value.length > 0 && (
                  <FileUploaderContent className="">
                    {field.value.map((file, i) => (
                      <FileUploaderItem
                        key={i}
                        index={i}
                        aria-roledescription={`file ${i + 1} containing ${
                          file.name
                        }`}
                        className="p-2 flex items-center"
                      >
                        <span>{file.name}</span>
                      </FileUploaderItem>
                    ))}
                  </FileUploaderContent>
                )}
              </FileUploader>
              <FormDescription>
                Upload an optional image to accompany your canvas.
              </FormDescription>
              <FormMessage />
            </FormItem>
          )}
        />

        {/* Canvas Name */}
        <FormField
          control={form.control}
          name="name"
          render={({ field }) => (
            <FormItem>
              <FormLabel>Canvas Name</FormLabel>
              <FormControl>
                <Input
                  placeholder="e.g. AI-powered dashboard"
                  disabled={isSubmitting}
                  {...field}
                />
              </FormControl>
              <FormMessage />
            </FormItem>
          )}
        />

        {/* Prompt */}
        <FormField
          control={form.control}
          name="prompt"
          render={({ field }) => (
            <FormItem>
              <FormLabel>Prompt</FormLabel>
              <FormControl>
                <Textarea
                  className="resize-none"
                  rows={4}
                  placeholder="Describe what you want to generate..."
                  disabled={isSubmitting}
                  {...field}
                />
              </FormControl>
              <FormDescription>
                A detailed prompt helps generate better and more accurate
                results.
              </FormDescription>
              <FormMessage />
            </FormItem>
          )}
        />

        <Button
          type="submit"
          className="w-full"
          variant="outline"
          disabled={isSubmitting}
        >
          {isSubmitting ? (
            <>
              <Loader2 className="mr-2 h-4 w-4 animate-spin" />
              {isUploading ? "Uploading..." : "Creating Canvas..."}
            </>
          ) : (
            "Create Canvas"
          )}
        </Button>
      </form>
    </Form>
  );
}
