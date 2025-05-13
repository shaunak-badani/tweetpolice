import * as React from "react";
import { cva, type VariantProps } from "class-variance-authority";
import { cn } from "@/lib/utils";

const backdropVariants = cva(
  "fixed inset-0 flex items-center justify-center transition-all",
  {
    variants: {
      variant: {
        default: "bg-background/80 backdrop-blur-sm",
        dim: "bg-black/50",
        blur: "backdrop-blur-sm bg-black/30",
      },
      fullscreen: {
        true: "z-50",
        false: "z-40",
      },
    },
    defaultVariants: {
      variant: "default",
      fullscreen: true,
    },
  }
);

export interface BackdropProps
  extends React.HTMLAttributes<HTMLDivElement>,
    VariantProps<typeof backdropVariants> {
  open?: boolean;
  onClose?: () => void;
  closeOnClick?: boolean;
}

const Backdrop = React.forwardRef<HTMLDivElement, BackdropProps>(
  (
    {
      className,
      variant,
      fullscreen,
      open = false,
      onClose,
      closeOnClick = true,
      children,
      ...props
    },
    ref
  ) => {
    if (!open) return null;

    const handleClick = (e: React.MouseEvent<HTMLDivElement>) => {
      if (e.target === e.currentTarget && closeOnClick && onClose) {
        onClose();
      }
    };

    return (
      <div
        ref={ref}
        className={cn(backdropVariants({ variant, fullscreen }), className)}
        onClick={handleClick}
        {...props}
      >
        {children}
      </div>
    );
  }
);

Backdrop.displayName = "Backdrop";

export { Backdrop, backdropVariants };