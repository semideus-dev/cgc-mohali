"use client";

import {
  useQuery,
  useMutation,
  useQueryClient,
  type QueryClient,
} from "@tanstack/react-query";
import { toast } from "sonner";

import {
  createCanvas,
  getCanvases,
  getCanvas,
  updateCanvas,
  deleteCanvas,
  type CreateCanvasInput,
  type UpdateCanvasInput,
} from "./actions";

// Types
export type Canvas = {
  id: string;
  name: string;
  url: string | null;
  prompt: string;
  userId: string;
  createdAt: Date;
  updatedAt: Date;
};

// Query keys
export const canvasKeys = {
  all: ["canvases"] as const,
  lists: () => [...canvasKeys.all, "list"] as const,
  list: (filters?: any) => [...canvasKeys.lists(), { filters }] as const,
  details: () => [...canvasKeys.all, "detail"] as const,
  detail: (id: string) => [...canvasKeys.details(), id] as const,
};

// Hook to get all canvases
export function useCanvases() {
  return useQuery({
    queryKey: canvasKeys.list(),
    queryFn: async () => {
      const result = await getCanvases();
      if (!result.success) {
        throw new Error(result.error);
      }
      return result.data;
    },
    staleTime: 5 * 60 * 1000, // 5 minutes
  });
}

// Hook to get single canvas
export function useCanvas(id: string) {
  return useQuery({
    queryKey: canvasKeys.detail(id),
    queryFn: async () => {
      const result = await getCanvas(id);
      if (!result.success) {
        throw new Error(result.error);
      }
      return result.data;
    },
    enabled: !!id,
    staleTime: 5 * 60 * 1000, // 5 minutes
  });
}

// Optimistic update helpers
function addOptimisticCanvas(
  queryClient: QueryClient,
  newCanvas: Omit<Canvas, "id" | "createdAt" | "updatedAt"> & { tempId: string }
) {
  queryClient.setQueryData<Canvas[]>(canvasKeys.list(), (old) => {
    if (!old) return [];

    const optimisticCanvas: Canvas = {
      id: newCanvas.tempId,
      name: newCanvas.name,
      url: newCanvas.url ?? null,
      prompt: newCanvas.prompt,
      userId: newCanvas.userId,
      createdAt: new Date(),
      updatedAt: new Date(),
    };

    return [optimisticCanvas, ...old];
  });
}

function updateOptimisticCanvas(
  queryClient: QueryClient,
  id: string,
  updates: Partial<Canvas>
) {
  // Update in list
  queryClient.setQueryData<Canvas[]>(canvasKeys.list(), (old) => {
    if (!old) return [];
    return old.map((canvas) =>
      canvas.id === id
        ? { ...canvas, ...updates, updatedAt: new Date() }
        : canvas
    );
  });

  // Update in detail cache
  queryClient.setQueryData<Canvas>(canvasKeys.detail(id), (old) => {
    if (!old) return old;
    return { ...old, ...updates, updatedAt: new Date() };
  });
}

function removeOptimisticCanvas(queryClient: QueryClient, id: string) {
  queryClient.setQueryData<Canvas[]>(canvasKeys.list(), (old) => {
    if (!old) return [];
    return old.filter((canvas) => canvas.id !== id);
  });
}

// Hook to create canvas
export function useCreateCanvas() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: createCanvas,
    onMutate: async (newCanvasData) => {
      // Cancel outgoing refetches
      await queryClient.cancelQueries({ queryKey: canvasKeys.list() });

      // Snapshot previous value
      const previousCanvases = queryClient.getQueryData<Canvas[]>(
        canvasKeys.list()
      );

      // Optimistically update
      const tempId = crypto.randomUUID();
      addOptimisticCanvas(queryClient, {
        ...newCanvasData,
        url: newCanvasData.url ?? null,
        tempId,
        userId: "temp-user-id", // This will be replaced by server response
      });

      return { previousCanvases, tempId };
    },
    onSuccess: (result, variables, context) => {
      if (result.success && result.data) {
        // Replace optimistic update with real data
        queryClient.setQueryData<Canvas[]>(canvasKeys.list(), (old) => {
          if (!old) return [result.data];

          const updatedList: Canvas[] = [];
          for (const canvas of old) {
            if (canvas.id === context?.tempId) {
              updatedList.push(result.data);
            } else {
              updatedList.push(canvas);
            }
          }
          return updatedList;
        });

        toast.success("Canvas created successfully!");
      } else {
        throw new Error(result.error || "Unknown error occurred");
      }
    },
    onError: (error, variables, context) => {
      // Restore previous state
      if (context?.previousCanvases) {
        queryClient.setQueryData(canvasKeys.list(), context.previousCanvases);
      }

      const errorMessage =
        error instanceof Error ? error.message : "Failed to create canvas";
      toast.error(errorMessage);
    },
    onSettled: () => {
      // Always refetch to ensure we have the latest data
      queryClient.invalidateQueries({ queryKey: canvasKeys.list() });
    },
  });
}

// Hook to update canvas
export function useUpdateCanvas() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: updateCanvas,
    onMutate: async (updateData) => {
      const { id, ...updates } = updateData;

      // Cancel outgoing refetches
      await queryClient.cancelQueries({ queryKey: canvasKeys.list() });
      await queryClient.cancelQueries({ queryKey: canvasKeys.detail(id) });

      // Snapshot previous values
      const previousCanvases = queryClient.getQueryData<Canvas[]>(
        canvasKeys.list()
      );
      const previousCanvas = queryClient.getQueryData<Canvas>(
        canvasKeys.detail(id)
      );

      // Optimistically update
      updateOptimisticCanvas(queryClient, id, updates);

      return { previousCanvases, previousCanvas, id };
    },
    onSuccess: (result, variables, context) => {
      if (result.success && result.data) {
        // Update with real data from server
        updateOptimisticCanvas(queryClient, context!.id, result.data);
        toast.success("Canvas updated successfully!");
      } else {
        throw new Error(result.error || "Unknown error occurred");
      }
    },
    onError: (error, variables, context) => {
      if (context?.previousCanvases) {
        queryClient.setQueryData(canvasKeys.list(), context.previousCanvases);
      }
      if (context?.previousCanvas) {
        queryClient.setQueryData(
          canvasKeys.detail(context.id),
          context.previousCanvas
        );
      }

      const errorMessage =
        error instanceof Error ? error.message : "Failed to update canvas";
      toast.error(errorMessage);
    },
    onSettled: (result, error, variables) => {
      // Refetch to ensure consistency
      queryClient.invalidateQueries({ queryKey: canvasKeys.list() });
      queryClient.invalidateQueries({
        queryKey: canvasKeys.detail(variables.id),
      });
    },
  });
}

// Hook to delete canvas
export function useDeleteCanvas() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: deleteCanvas,
    onMutate: async (id) => {
      // Cancel outgoing refetches
      await queryClient.cancelQueries({ queryKey: canvasKeys.list() });

      // Snapshot previous value
      const previousCanvases = queryClient.getQueryData<Canvas[]>(
        canvasKeys.list()
      );

      // Optimistically remove
      removeOptimisticCanvas(queryClient, id);

      return { previousCanvases, id };
    },
    onSuccess: (result, variables, context) => {
      if (result.success) {
        // Remove from detail cache
        queryClient.removeQueries({ queryKey: canvasKeys.detail(context!.id) });
        toast.success("Canvas deleted successfully!");
      } else {
        throw new Error(result.error);
      }
    },
    onError: (error, variables, context) => {
      // Restore previous state
      if (context?.previousCanvases) {
        queryClient.setQueryData(canvasKeys.list(), context.previousCanvases);
      }

      const errorMessage =
        error instanceof Error ? error.message : "Failed to delete canvas";
      toast.error(errorMessage);
    },
    onSettled: () => {
      // Always refetch to ensure we have the latest data
      queryClient.invalidateQueries({ queryKey: canvasKeys.list() });
    },
  });
}

// Utility hook for invalidating canvas queries
export function useInvalidateCanvases() {
  const queryClient = useQueryClient();

  return {
    invalidateAll: () =>
      queryClient.invalidateQueries({ queryKey: canvasKeys.all }),
    invalidateList: () =>
      queryClient.invalidateQueries({ queryKey: canvasKeys.list() }),
    invalidateDetail: (id: string) =>
      queryClient.invalidateQueries({ queryKey: canvasKeys.detail(id) }),
  };
}
