CDF       
      obs    -   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type                     	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?h�t�j~�   max       ?�bM���      �  �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       M��   max       P�?�      �  `   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       �u   max       =��m      �     effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>�\(�   max       @DУ�
=q       �   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ����R    max       @vd(�\       &�   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @%         max       @O            \  -�   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @�}        max       @���          �  .4   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       �#�
   max       >��      �  .�   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A�-u   max       B%�o      �  /�   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A���   max       B%�b      �  0P   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       >c��   max       C��X      �  1   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       >Z��   max       C��      �  1�   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max         }      �  2l   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          5      �  3    num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          #      �  3�   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       M��   max       P=�      �  4�   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?��!�.H�   max       ?�1���-�      �  5<   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       �u   max       >t�      �  5�   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>�\(�   max       @DУ�
=q       6�   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       �ٙ����    max       @vd(�\       =�   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @         max       @O            \  D�   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @�}        max       @���          �  E   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         ?�   max         ?�      �  E�   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?������   max       ?�($xG       Fx               M  }                        U         	      !                     A   \   #   
         
               8         	   
   
         NsƽN!tN��N�-+P��P�?�NnZ#N�%Ol	&O	�PN(�6N�(�N|3qO��N�s�NT��N��/O�aO�֪O�O�ҀO
�9O-Z�O��M��PJ�YP�HOed(N��6O��mO��OK-�O���O�46N�8vO-�P ̞OS3O!A�N���N�
�N��N6�7N�4�N�ș�u�e`B�#�
���
��o��o%   %   ;o;�o;��
<t�<t�<D��<T��<u<u<���<���<���<���<�/<��=C�=C�=\)=�P='�=,1=8Q�=D��=L��=L��=L��=aG�=e`B=ix�=�%=��=�7L=���=���=��=�F=��m !#&0<=CB<0#        ��������������������������������
#*//0/#[UU]k�������������m[# %6Ng����������gN6#()15BLMGB50)((((((((ZUZ[ht����}tphd[ZZZZ�~������������������������������������������������������������������������������������������������

���������������������������������������������������������������������
#')#
�����5-+,6BO[hnruuh[XPF85MJKNOX[hjqtxzxut[XOM������������������������������������������������������������XPOQU[fhhtt{�~yth[XX����������������������#/5BNSWTTNB5)
������������������������������������������������������������������))'# ����	"/:;?;9/"	&%)/5BN[\[XVYXUNB5)&MOOW[gt�������umg[NM;=DPanz�����znaUNLH;��������������������#/=HKHEJJH=<7#bet��������������zmb ���)5950)$  ��������������������99;>CHT[aegaa]YTRH;9nkknoz���|zxnnnnnnn
#/<BH</###
##)(#����������������((+/<AHNHG=<3/((((((�����������������������������������������T�[�Y�T�L�G�;�6�;�<�G�N�T�T�T�T�T�T�T�T���
������
������������������������D�EEEEEEEEED�D�D�D�D�D�D�D�D�D��g�������������������Z�A�*�����(�O�g��)�B�[�m�t�q�m�[�D�6�����������������	������	����������	�	�	�	�	�	�	�	���������������������������}������������āčĚĦīĭĪĦĚčā�k�h�]�_�h�n�t�wā�-�8�:�F�P�S�\�S�F�:�-�!������!�"�-�����������������������������������������{ǈǑǔǌǈ��{�o�h�b�[�^�b�o�r�{�{�{�{àçìù��������ùòìéãâàßààààDoD{D�D�D�D�D�D�D�D�D�D�D{DoDiDcDfDeDlDo�������	������ֹܹܹ�������������������������ù����������������������(�5�A�N�R�Z�R�N�A�@�5�(������������������������y�m�`�[�X�_�`�f�m�y�����������ʾѾܾ��ھʾ������]�E�Z�f�s��������(�*�4�=�9�4�(�#�����������������������ǿɿȿ����������~���������������A�M�Q�Z�f�s�z�����h�f�Z�M�A�-�)�4�@�A���������������������������������������������������ȼʼʼʼ������������������������ʼӼͼʼ�����������������������������ƎƳ���������������ƧƎ�p�Y�P�O�U�hƎ�Ϲ����������ܹù������x�z�}����Ź������������������������źŹŵűŹ�M�V�Z�]�`�Z�M�A�4�4�.�4�A�E�M�M�M�M�M�M�L�Y�r�~�������������~�r�e�Y�L�@�5�6�@�L�a�m�z������}�z�y�m�d�a�\�Z�T�S�T�X�a�a����������������ݿѿɿ˿ֿ���A�N�Z�g�k�v�|�|�{�s�g�Z�N�C�5�(�#�!�%�A���	��"�*�)�%�"�����������������������/�<�H�K�H�D�@�<�/�#��
���
��#�-�/�/¦¬²»¿º²ª¦�w������.�3�,�������ùæö����������������#�0�<�B�@�<�7�8�5�0�*�#���������(�4�;�D�D�A�;�4�.�(�����
�����U�b�h�n�{ńŎŔŗŔŇ�{�y�n�b�X�U�K�R�U�������������ܻۻܻ������������������������߽ݽݽݽ�������*�6�C�O�F�C�6�*�!�$�*�*�*�*�*�*�*�*�*�*EiEuE�E�E�E�E�E�E�E�E�EuEpEiEcE^EiEiEiEiĿ����������������ĿĳıĳĳĿĿĿĿĿĿ / i " 6 6  2 ,  4 7 1 G ( > H X N Q & ? Y S " [ J F > . D H - J . ^ 7 k F E m M | P 9 ,    �  f  �  �  ^  _  �  �  �  9  G  �  �  K  �  b    J    B  X  B  �  @    �  �  �  �  ,  Q  �    �    �  �  >  o  ?  �    G    ��#�
�o;��
<t�=��>��;�o;�`B<�<�t�<D��<�9X<�1=Ƨ�<���<�/<�j=t�=ix�='�=H�9=,1=0 �=m�h=t�=���>o=���=T��=�hs=}�=q��=�hs=�hs=��=���=�l�=��P=���=���=�{=�{=��>t�>VB%�oB/)B>ZB�$B �iB	UBrB�B[ZB�mB�sB�B!��BB )�B�B��B��B��B�_BzWB#CB*�B��B"��B�B&B��Bh�B��A�-uB�GB	��B�rBN(B6�B�B
SBV}A�"$B<6B� B��B�.B��B%�bBC�B@eB��B ��B	@�BE�B�GBMaB��B�B�^B!��B5�BڻB6BXEB��B�*B~:B�B"�UBt�B��B"}}B��B&cB/�BB�B��A���B��B	�YB�zB1BA�B;�B�B�A��B@�BJ%B��B�kBA�@�M{Ae;�A��C�\�A���AՊ$A�!Ar��A�kI@v��A�?yBA��C�Ų?-�hAό�A��Al�YAJ�!A3��As��A>�A���@�3�@�ġB��>c��A���A<c?�)�A��bA�K�A�ؔA��(A��2A�ВA�n�A���A6oA��@@��AA-��B K�C��XA�:`@��AeA��C�W�A�nWAԗ	A�}}Ar�AއH@z��A�|B��Ä́~C��>?4��AϐKA���Am	AK�A3QAt�/A=��A��3@�: @���BŁ>Z��A��aA;��?��yA���A�*�A���A���A���A���Aӆ4AꁮA6�?A�o@�A-PB <�C��A�z�               N  }                        V         	      !                     A   \   #            
               9         	      
                        5   3                                       %                     -   %                              )                                       #   #                                                            !                                                         NsƽN!tN��QN�-+O�+�P=�NnZ#N�5�O�.N�CN(�6N��NK�O�	N�s�NT��N��/O�aO�K_O"O[fO
�9O	�N��^M��O��O��(O��N��6O��mO��OK-�Ok{O�46N�8vO-�O�I?OS3O!A�N���N�
�N��N6�7N��N�ș    �  q  T  R  �  �  >  9  L    A  �  �  �    w  r  �  J  
  �  �  e  �  �  �  �    ,  �  �  �      L  �  �  �       |  �    l�u�e`B�t����
<�/>t�%   :�o<49X;��
;��
<#�
<#�
=,1<T��<u<u<���=o<���<�<�/=+=��=C�=m�h=}�=L��=,1=8Q�=D��=L��=P�`=L��=aG�=e`B=�t�=�%=��=�7L=���=���=��=�=��m !#&0<=CB<0#        ��������������������������������
#*//0/#wqnnqz�������������w=7<CN[gt�������t[ND=()15BLMGB50)((((((((W[^ht~|tihg\[WWWWWW��������������������������������������������������������������������������������������������������	

���������������������������������������������������������������������
#')#
�����549BO[^hlopph[POF>75NKLOZ[hiptwzwtph[YON������������������������������������������������������������TRUZ[hqty}{vthb[TTTT��������������������

)5BJLLKHC5)�����������������������������������������������������������������))'# ����	"/:;?;9/"	&%)/5BN[\[XVYXUNB5)&NPP[gt�������tkg[XPN;=DPanz�����znaUNLH;��������������������#/=HKHEJJH=<7#ywy~���������������y ���)5950)$  ��������������������99;>CHT[aegaa]YTRH;9nkknoz���|zxnnnnnnn
#/<BH</###
##)(#����������������((+/<AHNHG=<3/((((((�����������������������������������������T�[�Y�T�L�G�;�6�;�<�G�N�T�T�T�T�T�T�T�T���
������
������������������������D�EEEEEEEEED�D�D�D�D�D�D�D�D�D��Z�g���������������������Z�N�A�3�1�7�C�Z���6�B�O�W�Z�Y�S�H�6����������������	������	����������	�	�	�	�	�	�	�	����������������������������������������āčĚĦĦĨĦĤĚđčā�z�t�q�r�t�zāā�-�5�:�F�N�S�Z�S�F�:�7�-�!�����!�%�-�����������������������������������������{ǃǈǑǊǈ�}�{�o�j�b�\�^�b�o�v�{�{�{�{ìù��������ùïìëåäììììììììD{D�D�D�D�D�D�D�D�D�D�D�D�D{DzDqDsDzD{D{�������	������ֹܹܹ�������������������������ù����������������������(�5�A�N�R�Z�R�N�A�@�5�(������������������������y�m�`�[�X�_�`�f�m�y���������ʾ־۾ؾ׾̾��������q�f�s�t�����������(�4�8�7�4�(�"�������������� ����������ÿƿĿÿ������������������������A�M�Q�Z�f�s�z�����h�f�Z�M�A�-�)�4�@�A���������������������������������������������������Ƽ������������������������������ʼӼͼʼ�����������������������������ƎƚƧƳ������������ƳƚƎƁ�u�b�_�h�uƎ���Ϲܹ����������ܹϹ����������������������������������������������Ž���ƾM�V�Z�]�`�Z�M�A�4�4�.�4�A�E�M�M�M�M�M�M�L�Y�r�~�������������~�r�e�Y�L�@�5�6�@�L�a�m�z������}�z�y�m�d�a�\�Z�T�S�T�X�a�a����������������ݿѿɿ˿ֿ���A�N�Z�j�u�{�{�y�s�g�Z�N�D�5�(�$�'�(�5�A���	��"�*�)�%�"�����������������������/�<�H�K�H�D�@�<�/�#��
���
��#�-�/�/¦¬²»¿º²ª¦�w������&�+�-�$�����������������������#�0�<�B�@�<�7�8�5�0�*�#���������(�4�;�D�D�A�;�4�.�(�����
�����U�b�h�n�{ńŎŔŗŔŇ�{�y�n�b�X�U�K�R�U�������������ܻۻܻ������������������������߽ݽݽݽ�������*�6�C�O�F�C�6�*�!�$�*�*�*�*�*�*�*�*�*�*EuE�E�E�E�E�E�E�E�E�EuEpEiEdEiEiEuEuEuEuĿ����������������ĿĳıĳĳĿĿĿĿĿĿ / i ! 6 = 	 2 ( " 6 7 ) =  > H X N A  9 Y R  [ E @ 8 . D H - D . ^ 7 % F E m M | P - ,    �  f  �  �  G  O  �  �      G  �  v  O  �  b    J  2  )  �  B  K  �    �  p  <  �  ,  Q  �    �    �  �  >  o  ?  �    G  �  �  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�                         �  �  �  �  �  �  �  s  S  �  �  �                  �  �  �  �  �  �  �  �  �  o  q  n  h  ^  N  2    �  �  �  K    �  �  F  �  �  I  �  T  <  !    �  �  �  �  `  I  =  ;  2  $    �  �  �  �  \  �  �    {  �  �  /  L  L  ,    �  �  �  
  t  �  J    S  e  �  �  n  �  �    �  �  �  G  �  d  �  3      4  )  �  �  �  �  �  �  �  �  �  ~  t  j  ^  R  F  :  ,         �  9  ;  =  =  9  6  1  +  &          �  �  �  �  �  �  Y  j  �  �    *  6  9  .    �  �  �  u  D    �  x  �     �  A  I  I  D  8  '    �  �  �  �  �  n  R  8     	  �    Y                �  �  �  �  �  �  �  �  l  S  5    �  %  8  ?  <  3  %    �  �  �  �  w  L    �  �  �  D     �  ~  �  �  |  [  8    �  �  �  �  n  \  :  �  �  X    �  ^     �  &  �  �  �  �  �  �  �  �  2  �  /  x  
�  	�  	:  o    �  �  �  �  �  �  �  �  �  �  �  w  o  e  Y  N  6  �  �  5                   �  �  �  �  �  �  r  M  &  �  �  �  w  m  c  V  G  8  &      �  �  �  �  �  �  �  m  S  5    r  j  a  T  I  >  4  +  !    
  �  �  �  �  �  |  N    �  <  q  �  �  �  �  �  �  s  c  P  8    �  �  l    �  +  e  H  J  I  @  1       �  �  �  �  v  K    �  �  �  O  �  �  �  �      	    �  �  �  �  �  i  F  '    �  w  �  r   �  �  �  �  �  �  k  O  1    �  �  �  �  �  ]  6    �  �  ~  �  �  �  �  �  �  �  �  �  �  r  \  B  '    �  �  �  w  S  G  X  a  e  b  U  E  2      �  �  �  B  �  �  /  �  B  �  �  �  }  z  w  s  p  m  j  f  a  Y  Q  J  B  :  2  +  #    �  �  ;  m  �  �  �  �  �  �  P    �  D  �  E  �  �  �  i  
�  (  �  �  �  �  �  �  �  l  .  
�  
�  
+  	�  �        �  B  c  �  �  �  �  �  �  Y  "  �  �  k  (  �  �  \    �          �  �  �  �  �  �  �  v  a  J  2    �  �  4  �  �  ,  $            �  �  �  Y     �  �  L  �  �  x  q  x  �  �  �  �  �  �  v  V  ?  %  
  �  �  �  �  �  �  n  S  8  �  �  �  �  �  �  �  �  z  e  O  7      �  �  �  �  l  G  �  �  �  �  p  Y  B  (       �  �  �  g  4  �  �  t  )  y        �  �  �  �  �  �  m  L  /      �  �  i    h   �    	  �  �  �  �  �  �  k  S  4    �  �  s  :  �  �  z  >  L    �  �  U    �     �  �  �  j  9     �  �  g  2    �  �  �  �  �  w  �  �  �  v  D    �  h    �    �  �  g    �  �  �  x  r  n  h  `  S  C  -    �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  {  i  S  :    �  �  �  Y  .    �  �  �  r  Q  :  "    	  �  �  �  �  �  w  ^  (  �  ]        �  �  �  �  p  W  =  #    �  �  �  �  \  -  �  �  �  |  ;  �  �  �  �  �  �  �  �  �  �  �  �  �  �  W    �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  v  l  ^  P  A  �    �  �  �  �  t  >  �  �  f    �  Y  �  �    �    �  l    �  �  �  �  {  R  %  �  �  �  �  �  ~  q  p  ~  �  �