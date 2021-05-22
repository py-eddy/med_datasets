CDF       
      obs    4   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type                     	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?`bM���   max       ?��t�j      �  �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       MŐ}   max       PP�S      �  |   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       ����   max       =�      �  L   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>��Q�   max       @FQ��            effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?�G�z�    max       @vy��R        (<   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @"         max       @Q�           h  0\   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @��        max       @�n�          �  0�   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       ��t�   max       >��      �  1�   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A�W   max       B/�      �  2d   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A���   max       B/��      �  34   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       ?Lt�   max       C���      �  4   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       ?O<�   max       C���      �  4�   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          �      �  5�   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          1      �  6t   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          1      �  7D   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       MŐ}   max       PH�,      �  8   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?��n.��   max       ?���@��      �  8�   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       ����   max       >	7L      �  9�   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>��Q�   max       @FQ��        :�   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?�G�z�    max       @vy��R        B�   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @!         max       @Q�           h  J�   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @��        max       @�n�          �  K,   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         E�   max         E�      �  K�   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�X�e+�   max       ?���)^�     @  L�      *               &   )         	                  �               )      (      F   b      &                  7   q         7   &               	   =   *             �   N���O�=�N.�YN��QO9MŐ}O�cO��-O&"M�u>N��"O8�N���N%��O0\�Or��PH�,O�Q0N/�N��hM���O�$�O�a P)	�O��O˓�PP�SO���O���O<0�O7��NB�pN�M+OL�O��PqN���O �P��O��Nr?�N��O<�NǜN��O��O��5N�/wO�64N~F.O�i<ND6�����t��o;o;��
;�`B<o<t�<t�<T��<T��<e`B<u<��
<ě�<���<�/<��=+=\)=t�=��='�=8Q�=@�=@�=P�`=P�`=P�`=aG�=aG�=aG�=e`B=ix�=ix�=ix�=y�#=}�=}�=}�=}�=�7L=��P=��P=��T=���=�Q�=�v�=��=��=��=�~���������������~~~~��������������������!)45666/)')3/*)255:<BHU]afihfaYUH<2�������������������������
#/497/#����� #/<HanvpUH?</+# ��������������������"*(&" ����������������������������������������#&(+//7<HIKKHH@<1/##faahjsty���trhffffff���������

�������##0<INUXSOJI<0&#��������������������SNNSaz����������zmaS&)6BOOQOB64)&&&&&&&&#0<EA<<40&#��������������������������
)980#
�����
#/<UXaekfaU</#
������)4BD>8)���)6O[aec[OB6&�����
#0<F<$
������������� ��������������� �����������������������
)+58BHWNB5)"�������
"$#!
����������������������*46>CCC6*"/:;HNTZ]_XTH;/"]^m�����������zmfba]'',3BNg������|g[B5+'�����������������������������������6BIJE=6)���		)69>BIIGB<61	'*6A>86*xxz~��������}zxxxxxx�����������LCN[\`\[SNLLLLLLLLLL
##).#
���������

����������"%$$"���#-/10/#
		
#####�������������������	))-*)�������� 

����+-/<EHA<//++++++++++�������������������������(�5�A�Z�g�������w�A�5�(�$�2�1�)����(ÇÓÛàäàÓÐÇÆÇÉÇÀÇÇÇÇÇÇ�����
�� �!� ��
�����������������������������������������������������z�u�r�x��čĎĐĚěĚčāĀāĉČčččččččč��(�A�K�V�W�V�P�H�A�(��������������������	������������������������������
�
�������������������������T�a�a�d�a�T�H�A�H�P�T�T�T�T�T�T�T�T�T�T�#�/�7�<�=�<�7�0�/�)�#����#�#�#�#�#�#���'�1�4�:�@�D�B�@�4�'��	���������D�D�D�EEEEE#EEEED�D�D�D�D�D�D�D߻_�l�x�{�x�l�h�_�V�S�N�L�S�\�_�_�_�_�_�_���������������������������z�x�r�i�m��r�����������������������r�f�d�\�Y�f�r�y�����ݾ�(�@�E�F�C�1�(�����н������y����������
���	�����������������������������������z�y�w�y�z�������������������ּ���������������޼ּӼּּ����
����
�����������������������������;�T�`�y�������y�`�T�G�@�>�6���ھ���
��"�)�(�������������������������
��(�5�=�B�A�K�M�J�A�(�����޿ԿԿ߿�����ʾ۾�����׾ʾ�����������������������!�(�/�1�-�#��������Ѻ˺Ǻֺ̺��������������������ìÇ�{�q�o�o�l�xàø���B�N�[�t�t�b�N�E�B�;�9�:�B�y���������������������y�l�`�S�M�S�`�l�y�����������������������y�j�e�b�m�y������(�4�A�M�Z�f�i�w�s�g�f�Z�M�A�4�(���#�(���ʼּټּԼʼ��������������������������y���������������{�y�x�m�o�w�y�y�y�y�y�y�#�0�<�I�M�U�V�Z�U�Q�I�<�0�#�!���� �#������� ������������ŽųŬŤţŬŹ�������������'�+�-�+��������������������ÓàìùûùìëàÔÓÇÆÄÇÉÓÓÓÓ���������'�2�'������������r�����ĺɺ˺ԺȺ������r�f�W�L�@�6�I�Y�r���!�3�E�M�I�F�:�!�������ٺ�����S�`�e�l�n�r�l�`�S�R�N�N�S�S�S�S�S�S�S�SFFF F$F*F$F#FFE�E�E�E�FFFFFFF���������������������������������������������������������������������������������g�t�|�t�s�g�g�[�N�G�N�N�[�]�g�g�g�g�g�gEuE�E�E�E�E�E�E�E�E�E�E�E�E�E�ExErEnEkEu�0�<�I�U�b�d�k�k�Z�I�0�#��
������"�0ǭǡǛǔǈǅǀǄǈǒǔǡǣǭǯǭǭǭǭǭ�ûлܻ��������������лû������ü��'�'�4�6�4�3�'�����
������D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D����!�(�����������������  K B D " � h F , E W 1 [ � S " D P t C o u F " ; : / ) : l @ 1 > - / R ` V 9 3 l = N x f L 7 ) 5 Y  -  �  D  K  �  |  T  P  R  k  !  �  �    n  �  �  �  ,  �  %  I  J  u  �  �  �  �    F  �  �  ^  �  �  	  �  �  '  �  N  �  �  �  	  �  i  b  �     �    O��t�=+���
;��
<�C�<#�
=8Q�=H�9<�`B<u<�1<�h=��<���=C�=8Q�<�/=P�`=t�=D��=��=���=��=�{=��w=>�u=��P=�9X=u=�C�=u=�%=�t�=�S�>-V=��T=��
=�h=���=�O�=�Q�=��=���=�E�>n�>$�=�"�>%=���>��>%B}�B��B%�B;�Bw�BIJB5�BփB��A�WBݣB!b�B��B=�B"�lB&B!�A��JB�B%9�B�B��B�Bj�B��B#�4B�kBB,8(B��B�;Bd�B/�A���B ��B�=B"��BZ/B��BN/B/�B۲B�SB��BUB:\Bw�B��B/�B�KB�<B��B��B��B-�Bw�B�B�^B<�B��B̌A���B��B!H|B��B�B"O<B&9oB!�bA���B�B%?B��B?bB�nB�GB@�B#�B�EB�B,<RB:5B?�BB�B/�ZA�	�BE�B�TB"@yBFtB�wB%�B/��B�$B��B@SBL^B��BvvB��B��B�8B��B�?Lt�A��A�G�A�XHA��}A�}�A�?A�}�AЕ�A��{A��:@ŕ�C�P�@�!�@��@�,A,�5A�A�d,A �A�-�Afb'A���A���AO��@S��A���A�8<A;�Ao۵A<�@�!�An)�A�	RA��xA�W�Aˆ�?]��@�A@e��Ab�C���A��OA��A�C�"A��SBXa@�!@�C��A���?P=�A�y�A���A�u�A��mAޓbA�~�A�{�AЅ�A���Ac@��C�P;@��@��i@��A,ݎA���A�o�A�A��pAi �A��$A�bEAO�@S��A���A��A6Ar��A:�A@��jAn��A�|sA���A�}gA˘"?O<�@��@eo�A��C���A��A�AA���C�)cA���B?�@�C@��C��A���   	   +               &   )         	                                 *      )       G   c      &                  7   q         7   &               	   =   *      !      �         '               !                              1   )            +      )      %   -      !                     '         )                                                            !                              1   )                  #         !                                    %                                       N���O)�QN.�YN��QN���MŐ}O�cN�ɲNA�M�u>NW�N�a%N�E_N%��O$Or��PH�,O�Q0N/�N�r�M���O���O�a PǭO�>�OsKO��O���N��O<0�O#��NB�pN�M+OL�O�&HO�\N��cO �O���Ou
�Nr?�N��O<�NǜN��Ov�bO��5N�/wO�64N~F.OQ'�ND6  �  5  &  �  �  �  :  n  J  G  (  �  	  �  �  �  	�  �  #  �  �  �  �  Y  	  �    0  R  �  �    �  �  	�  !    �  g    �  	3  /  �  �  �  	F  Y    �  ^  ����<49X�o;o<t�;�`B<o<�/<�t�<T��<e`B<���<�o<��
<���<���<�/<��=+=��=t�=49X='�=P�`=L��=�7L=��=P�`=�+=aG�=e`B=aG�=e`B=ix�=��=��
=}�=}�=�7L=�7L=}�=�7L=��P=��P=��T=�Q�=�Q�=�v�=��=��>	7L=�~���������������~~~~��������������������!)45666/)')3/*)9<?HU]acba]ULHF<<<99�������������������������
#/497/#�����+**/8<=HNUUUPH><8/++��������������������"*(&" ����������������������������������������(&)+//8<GHJKHH@<0/((faahjsty���trhffffff��������

��������##0<INUXSOJI<0&#��������������������SNNSaz����������zmaS&)6BOOQOB64)&&&&&&&&#00700#���������������������������
%/53)#
���
#/<UXaekfaU</#
�����)5<>7)�����)6O[^cca[OB6)�������
&(#
��������������������������� �����������������������
)+58BHWNB5)"�����
!#$#" 
�����������������������*46>CCC6*"/:;HNTZ]_XTH;/"gggmw������������zmg1/03=Ngt�����~g[NB51������������������������������������6BEHC;6)��)6;@DFFB6)%'*6A>86*xxz~��������}zxxxxxx�����������LCN[\`\[SNLLLLLLLLLL
##).#
��������

����������"%$$"���#-/10/#
		
#####�������������������	))-*)��������

�����+-/<EHA<//++++++++++�������������������������(�5�A�N�Z�_�g�j�o�l�g�Z�N�A�:�5�-�#�&�(ÇÓÛàäàÓÐÇÆÇÉÇÀÇÇÇÇÇÇ�����
�� �!� ��
���������������������������������������������~�z�x�z����������čĎĐĚěĚčāĀāĉČčččččččč��(�A�K�V�W�V�P�H�A�(�������������������	����������������������������������������������������������������T�a�a�d�a�T�H�A�H�P�T�T�T�T�T�T�T�T�T�T�#�/�5�<�<�<�6�/�/�.�#����#�#�#�#�#�#���'�)�0�4�'�"�������������D�D�D�EEEEE"EEEED�D�D�D�D�D�D�D߻_�l�x�{�x�l�h�_�V�S�N�L�S�\�_�_�_�_�_�_���������������������������{�z�r�k�n��r�����������������������r�f�d�\�Y�f�r�y�����ݾ�(�@�E�F�C�1�(�����н������y����������
���	�����������������������������������z�y�w�y�z���������������������������������ؼ޼�����������
����
�����������������������������"�;�T�m�y�������y�m�`�T�G�D�B�=�&�"��
��"�)�(�������������������������
��(�5�:�;�A�F�D�5�(�����޿��������ʾ׾����ܾ׾ʾ���������������������������$�%��������ںԺӺֺܺ�������������������ùìÜÓÌÈÌ×çù���B�N�[�t�t�b�N�E�B�;�9�:�B���������������������������y�x�n�r�y�z�������������������������y�j�e�b�m�y������4�A�M�Z�f�t�s�f�Z�M�F�A�8�4�(���(�+�4���ʼּټּԼʼ��������������������������y���������������{�y�x�m�o�w�y�y�y�y�y�y�#�0�<�I�M�U�V�Z�U�Q�I�<�0�#�!���� �#����������������������������ŶŰŪŬŸ����������%�'�&�#�������������������ÓàìùûùìêàÓÇÅÇÊÓÓÓÓÓÓ���������'�2�'������������r�~�������źƺ˺º������r�i�\�Q�N�L�Q�r���!�,�:�A�J�F�:�!��������������S�`�e�l�n�r�l�`�S�R�N�N�S�S�S�S�S�S�S�SFFF F$F*F$F#FFE�E�E�E�FFFFFFF���������������������������������������������������������������������������������g�t�|�t�s�g�g�[�N�G�N�N�[�]�g�g�g�g�g�gE�E�E�E�E�E�E�E�E�E�E�E�E�E�E�EzEuErEtE��0�<�I�U�b�d�k�k�Z�I�0�#��
������"�0ǭǡǛǔǈǅǀǄǈǒǔǡǣǭǯǭǭǭǭǭ�ûлܻ��������������лû������ü��'�'�4�6�4�3�'�����
������D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D����!�(�����������������  R B D   � h  , E X # X � S " D P t $ o N F ( 4 $ 0 ) / l A 1 > - 1 ; ] V 4 / l = N x f L 7 ) 5 Y  -  �  �  K  �  �  T  P  �  T  !  �  �    n  z  �  �  ,  �    I  c  u  J  Z  �  !    %  �  g  ^  �  �  �  �  �  '  4  �  �  �  �  	  �  	  b  �     �  �  O  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  �  �  �  �  �  �  �  �  �  ~  k  X  D  -    �  �  �  �  �  F  ]  x  �  �    &  ,  1  5  1     �  �  u  #  �  ?  O  2  &  !              �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  {  s  j  a  Y  x  �  �  �  �  �  �  �  �  �  �  r  Y  =    �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  :  &    �  �  �  �  c  C  '    	  �  �  �  �    �  +  �  �  >  �  �  
  .  H  _  m  i  X  9    �  �     �    o  �  �  �  �  �  �      ;  J  J  @  0      �  �  �  F  �  �  G  A  ;  5  /  )  #                       #  %  (    "  &  %  !        �  �  �  }  R  $  �  �  �  Z     �  �  �  �  �  �  �  �  �  �  �  �  s  `  M  ;  &  
  �  �  S  �  �  �  �  �  i  3  �  �  �  J  
  �  �  F    �  �  ?  �  �  �  �  �  �  �  �  �  g  8    �  �  �  �  m  K  '     �  �  �  �  �  �  �  �  �  �  �  m  V  :    �  �  �  �  �  o  �  �  �  t  _  =    �  �  �  �  �  �  �  �  �  l  w  d  M  	�  	�  	�  	�  	�  	�  	�  	�  	�  	�  	�  	�  	�  	�  	�  	�  	�  	�  	�  	�  �  �  �  �  z  K    �  �  �  �  d  5     �  �  <  �  r   �  #          �  �  �  �  �  �  �  �  u  b  P  =  +      �  �  �  �  �  �  �  �  �  �  �  �  r  M  +    �  �  �  �  �  �  }  w  q  k  e  _  Y  S  L  D  =  5  -  &          �  �  �  �  �  m  :    �  �  �  [  (  �  �  A  �  T  �  g  �  �  �  �  �  �  s  ^  7    �  �  w  ;    �  �  d  e  �    ?  T  Y  V  O  C  1    �  �  �  u  B  �  �  M  �  N    �        �  �  �  �  �  c  8    �  �  X    �  H  �   �  �  \  �  �  �  �  �  �  �  �  S    �  N  �  0  �  �  �    	�  
]  
�  
�  
�        
�  
�  
�  
u  
  	�  	!  v  �  �  #  �  0  *      �  �  �  �  �  �  f  H  +  
  �  �  |  F    �  �  -  �  �    0  G  P  Q  C  +    �  �  y  '  �  $  t  �  �  �  �  o  V  E  4  #            �  �  �  �  �  �  �  �  �  �  �  �  {  m  ^  O  ?  -    �  �  �  q  ?  �  �  �              �  �  �  �  �  �  �  �  �  w  d  P  =  )  �  �  �  }  k  Y  E  /       �  �  �  �  p  V  ;     �   �  �  �  W  6    �  �  �  y  X  ;  '    �  �  �  �  r  E    	  	{  	�  	�  	�  	{  	f  	E  	  �  �    �  ]    �    ^  �  �  �  �  �    !    �  �  �  7  �  �    �  
  
Y  	r  �  &  �  �      �  �  �  Z     �  �  v  <  �  �  �  s  &  �  �  �  �  �  g  I  %  �  �  �  q  C    �  �  �  �  �  �  ~  1    "  M  f  Y  U  \  \  8    �  �  g    �  �  z  5  �    �  �          �  �  �  �  u  E  
  �  x    �  7  �  �  �  �  �  ~  k  T  >  (    �  �  �  �  �  n  G    �  �  X    	3  	  �  �  �  �  J    �  L  �  v  �  �    �  �  p  �  c  /    �  �  �  �  �  n  L  )  �  �  �  J  �  p  �  t  �  r  �  �  �  �  �  �  �  �  �  �  y  c  R  D  6  (    
  �  �  �  �  �  �  s  [  C  ,    �  �  �  �  c  ?    �  �  �  r  �  �  �  �  �  �  �  G  �  �    
�  	�  	F  �  �  �  �  �  _  	F  	<  	.  	  	  �  �  �  r  =  �  �  >  �  !  �    �  �  d  Y  2    �  �  �  �  k  ?    �  �  ~  B  �  �  "  �     r    �  �  �  �  �  e  B    �  �  t    �  M  �  d  �     �  �  �  �  x  i  Y  I  5      �  �  �  �    d  H  .    �  �    =  V  ]  Q  /  �  �  3  �    3  #  �      8  	�  M      �  �  �  �  b  ?    �  �  �  \    �  u  ,  �  �  a