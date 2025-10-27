"use client";

import { useState } from "react";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog";
import { Plus } from "lucide-react";
import CanvasForm from "./canvas-form";

export default function CanvasDialog() {
  const [isOpen, setIsOpen] = useState(false);

  const handleFormSuccess = () => {
    setIsOpen(false);
  };

  return (
    <Dialog open={isOpen} onOpenChange={setIsOpen}>
      <DialogTrigger className="flex items-center flex-col gap-2 hover:bg-border/20 transition-all cursor-pointer text-muted-foreground justify-center border-2 rounded-lg p-4 border-dashed">
        <Plus size={36} />
        <span className="text-2xl">New Canvas</span>
      </DialogTrigger>
      <DialogContent className="max-w-4xl max-h-[90vh] overflow-y-auto">
        <DialogHeader>
          <DialogTitle>New Canvas</DialogTitle>
          <DialogDescription>
            Upload a banner/poster to get a critque report.
          </DialogDescription>
        </DialogHeader>
        <CanvasForm onSuccess={handleFormSuccess} />
      </DialogContent>
    </Dialog>
  );
}
