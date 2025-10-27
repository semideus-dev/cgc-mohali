"use client";

import { Plus, Image, Calendar, ExternalLink } from "lucide-react";
import CanvasDialog from "./canvas-dialog";
import { useCanvases } from "@/modules/canvas/server/hooks";
import {
  Card,
  CardContent,
  CardFooter,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";

function CanvasCard({ canvas }: { canvas: any }) {
  const formatDate = (date: Date) => {
    return new Date(date).toLocaleDateString("en-US", {
      month: "short",
      day: "numeric",
      year: "numeric",
    });
  };

  return (
    <Card className="group hover:shadow-lg transition-all duration-300 h-fit flex flex-col">
      <CardHeader className="pb-3">
          <CardTitle className="flex items-center justify-between" title={canvas.name}>
            <span className="flex text-xl">{canvas.name}</span>
            <div className="flex items-center gap-2 text-xs text-muted-foreground">
              <Calendar className="h-3 w-3" />
              {formatDate(canvas.createdAt)}
            </div>
          </CardTitle>
          {/* {canvas.url && (
            <Button
              variant="ghost"
              size="sm"
              className="opacity-0 group-hover:opacity-100 transition-opacity"
              onClick={() => window.open(canvas.url, "_blank")}
            >
              <ExternalLink className="h-4 w-4" />
            </Button>
          )} */}
      </CardHeader>

      <CardContent className="flex-1 pb-3">
        {canvas.url ? (
          <div className="aspect-video bg-muted rounded-md mb-3 overflow-hidden">
            <img
              src={canvas.url}
              alt={canvas.name}
              className="w-full h-full object-cover"
              onError={(e) => {
                const target = e.target as HTMLImageElement;
                target.style.display = "none";
                target.nextElementSibling?.classList.remove("hidden");
              }}
            />
            <div className="w-full h-full hidden items-center justify-center bg-muted">
              <Image className="h-8 w-8 text-muted-foreground" />
            </div>
          </div>
        ) : (
          <div className="aspect-video bg-muted rounded-md mb-3 flex items-center justify-center">
            <Image className="h-8 w-8 text-muted-foreground" />
          </div>
        )}
      </CardContent>
    </Card>
  );
}

function CanvasCardSkeleton() {
  return (
    <Card className="h-[250px] flex flex-col">
      <CardHeader className="pb-3">
        <Skeleton className="h-6 w-3/4" />
      </CardHeader>
      <CardContent className="flex-1 pb-3">
        <Skeleton className="aspect-video w-full rounded-md mb-3" />
        <Skeleton className="h-4 w-full mb-1" />
        <Skeleton className="h-4 w-2/3" />
      </CardContent>
      <CardFooter className="pt-0 pb-4">
        <div className="flex items-center justify-between w-full">
          <Skeleton className="h-4 w-20" />
          <Skeleton className="h-5 w-16" />
        </div>
      </CardFooter>
    </Card>
  );
}

export default function CanvasGrid() {
  const { data: canvases, isLoading, error } = useCanvases();

  if (error) {
    return (
      <div className="flex w-[90%]">
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 w-full">
          <CanvasDialog />
          <Card className="h-[250px] flex items-center justify-center">
            <div className="text-center">
              <p className="text-red-500 mb-2">Failed to load canvases</p>
              <Button
                variant="outline"
                onClick={() => window.location.reload()}
              >
                Retry
              </Button>
            </div>
          </Card>
        </div>
      </div>
    );
  }

  return (
    <div className="flex w-[90%]">
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4 w-full">
        <CanvasDialog />

        {isLoading ? (
          // Show skeleton loading cards
          <>
            {Array.from({ length: 5 }).map((_, i) => (
              <CanvasCardSkeleton key={i} />
            ))}
          </>
        ) : canvases && canvases.length > 0 ? (
          // Show actual canvases
          canvases.map((canvas) => (
            <CanvasCard key={canvas.id} canvas={canvas} />
          ))
        ) : (
          // Show empty state
          <Card className="h-[250px] flex items-center justify-center">
            <div className="text-center text-muted-foreground">
              <Image className="h-12 w-12 mx-auto mb-2 opacity-50" />
              <p>No canvases yet</p>
              <p className="text-sm">Create your first canvas to get started</p>
            </div>
          </Card>
        )}
      </div>
    </div>
  );
}
