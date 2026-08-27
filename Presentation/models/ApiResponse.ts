/**
 * : Success / Message / Code / AdditionalData / CarryOnData.
 */
export interface ApiResponse<TData = unknown> {
  Success: boolean;
  Message?: string;
  Code?: string | number;
  AdditionalData?: TData;
  CarryOnData?: unknown;
}
