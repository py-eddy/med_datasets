CDF       
      obs    5   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type                     	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?h�t�j~�   max       ?��l�C��      �  �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       N�h   max       Q?D      �  �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       ��`B   max       >         �  T   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>���R   max       @F           H   (   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ��(�\    max       @vn�Q�     H  (p   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @&         max       @Q�           l  0�   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @�L        max       @���          �  1$   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       �o   max       >��F      �  1�   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A��   max       B/��      �  2�   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A���   max       B/�2      �  3�   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       =�A?   max       C�k      �  4t   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       > �
   max       C��      �  5H   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max         8      �  6   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          O      �  6�   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          9      �  7�   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       N�h   max       P�ޮ      �  8�   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�MjOw   max       ?��Q�      �  9l   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       �ě�   max       >333      �  :@   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>���R   max       @F           H  ;   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ��=p��
    max       @vk
=p��     H  C\   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @         max       @Q�           l  K�   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @�L        max       @���          �  L   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         F�   max         F�      �  L�   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?��*0U2b   max       ?�����     �  M�   '   	               -               
   
   ^      &      '      1      #      +      $       	   	   `     7      #      I         )      '      *         !                  {   O���N��O#�O�qyN`.�O�S�PH�mN�	O���O��N�D(O]��O���Q?DN���OMPbO�%	O���NQo�P)`N�*EO��6N�`O��N� �O�sjO�&�NCQ�N�2aPP�HOu�P_"�N��Oz�N�I�O�mtO�a�N|ÈO�%�N0�O΋lP�RO��N-��O��O�ANB�*N$�N�TO7�N�hOpN�NIM��`B�ě����
��o��o:�o;D��<o<o<#�
<D��<T��<e`B<�o<�C�<��
<�j<���<�/<�h<�h<�<�=o=+=+=C�=\)=�P=��=��=#�
=0 �=49X=8Q�=8Q�=8Q�=<j=H�9=H�9=L��=P�`=aG�=ix�=m�h=m�h=q��=�%=�+=�O�=��P=��T>   ��������������������
#./850/#

��������������������'/<HUamynkkh`UH<'���������������

"/19AFHKH;/"KHOLV[h����������tSK)5>85)��������������������TSU[g������������tmT##+/<@GC=</.$#######-<E<<OSXfijgjfbUI<0-�����������������������5PZaqvrg)�����
 )/33.)!"#/<HU\\`a]UH</&#!����������������UW]gt�����������rl^U��������������������c^[g��������������tc�����
 #''#
���������"--%����##�������%�������
)12,)%���������  ��������~{�����������������~��������������������������������������������)5BMMH5-�������������

����1-2;[gt�������tgNB71��������������������[VFB64-6BNY[_[�����������������������
#),2<DFD</+�	#/>AJTZUH</
	���������������������~������������������*/1+*'��������������������������������������������'-2.0)���Z[ahrt{trha[ZZZZZZZZ��)5=@NUZNB5)|{����������������|��������������������('"6CFIC@6**66666���������	

���

 









�������	 �������)15765)'�����ùйܹ�������Ĺ������������������A�B�N�N�W�N�M�A�5�(�'�"�(�(�5�9�A�A�A�A���������úúº��������������������������������������������������������|������������������������������������������������	��"�;�H�V�W�M�H�;�/�"��	����������	���A�M�s�}���������s�Z�A�(���������A�M�Z�b�e�f�Z�M�H�C�A�@�A�A�A�A�A�A�A�A�׾����"�*�.�(�&�"��	� ���׾ʾ����׾�������������¾�����������s�g�f�m�s��y�y���������y�m�`�`�`�`�m�y�y�y�y�y�y�y�4�<�M�f�����������r�f�Y�M�;�*�(�.�1�4�T�a�h�m�m�g�_�T�F�;�/�"����"�/�;�H�T�����#�����Ɓ�\�C�*�������*�a�uƎƳ��¦±²´²­¦�z�x��������� �#������������������������ûлܻ���	�����л����������������3�@�L�Y�c�f�`�Y�@�3�'��������'�3�a�h�n�q�n�n�g�a�[�V�U�S�U�]�a�a�a�a�a�a������.�?�>�7�5�2�/�"�	�����������������;�D�H�T�W�Y�T�Q�H�;�7�/�"��"�%�.�/�0�;��;�G�`�k�m�x�n�`�T�"�	���߾����	��������������y�y�y�{���������������������)�B�]�k�h�_�R�N�C�B�5������
���)�(�5�A�N�Q�Z�\�^�Z�N�A�<�5�-�(�(�(�(�(�(���Ľݽ���"�$�����Ľ������z��������ù����������������������þùëëçëíù�<�H�T�Q�H�<�/�/�/�0�<�<�<�<�<�<�<�<�<�<�a�c�m�r�z�m�a�T�T�H�A�H�T�W�a�a�a�a�a�a��������������������������z�y�~�������׾����	������	����ؾʾ������ʾ�����)�6�L�N�K�=�)���������ùøý�������A�B�M�Y�Z�a�\�Z�M�A�9�4�2�4�4�;�A�A�A�A���ɺºɺֺ���������"�����������������������������������E�E�E�E�E�E�E�E�E�E�E�EiE\ELEJEPE\EiE�E��Z�f����������s�M�(���	�����(�4�ZÓÝàêåäàÓÊÊÊËÓÓÓÓÓÓÓÓ�����ɺغ��������ֺɺ��������������y���������y�l�l�l�q�y�y�y�y�y�y�y�y�y�y�нĽ��������y�w�n�`�l�o�y���������ӽϽ��g�����������������������s�g�A�1�0�A�N�g�r�~�����������������~�e�L�@�L�R�S�V�f�r�r�w�����~�r�g�f�^�f�n�r�r�r�r�r�r�r�r���������������������|�y�m�Z�R�M�R�`�m���������Ŀ�����(�*�����ѿ������������������������������������������������������������ ������������������y�o�m�i�g�i�m�y�������������������#�/�<�C�H�U�a�a�e�a�U�T�H�<�/�*�#���#ǺǭǧǡǔǒǑǔǡǭǺǺǺǺǺǺǺǺǺǺD�D�D�D�D�D�D�D�D�D�D�D�D{DvDoDfD{D�D�D��������
���������
�
������� 5 - $ 1 1 4 = A 2 7  ^ C S  * ^ & b @ U C E 8 L Y  . P - 6  8 B ] Y b W 7 / 4 a - 0 C m  6 D 3 f 6 �    (  �  c  R  j  #  `  �  �  �  �    �  	.    �  �  ?  �  �  C  G  @  �  �  +    \  �  }  �  ~  �  "  �  K  �  �    ,  �    �  E  �  �  M  3  �  >  \  �  �=o;D��<�t�<����o<�/=D��<e`B<ě�<���<���<�j<ě�=�G�<�/=ix�=P�`=�o=49X=��w=t�=��=\)=��P=49X=�C�=�+=49X=8Q�>1'=�7L>��F=]/=���=@�==���=e`B=�Q�=T��=�9X=��=Ƨ�=�%=��=�^5=�+=�C�=��w=�E�=��T>O�;>$�B��B�*B!�B%�B=�A��B�B��B!^�B
f;Bt�B&�`B�[Bv�BC�BܖB#�B� Bz�BeeB��B�Bi�B�xB��B"�NB��Bd]B��B�1BB	<4B,B�BE�BϥB,nB!��BR'B/}B+�B��B�B��B�#B�!BωB�RB/��B
B�)B�}B>DB�B��B!�5B=uBE�A���B��B�B!PB
6fB@^B'@$B��BA�B@_B��B"�qB�ABI�B
C�B�MB JB��B8uB�$B"�vB�`B�MB£B��B=�B	?jBA�B�WB:�B�QB��B"�B��B/��B+��B{yB�=B�jBE�B>�B�eBS%B/�2B;�B� B��B��=�A?A��@p�A�#FA�x�A��gA5�A>sAW�dAG�Al=R@��kA��NB�RA�?�A��@��{?�AƘ�A��A�-�A_��Ao_A��[A��#A)��A�#nA���A�CgA�|�AV�Aӓ�A<�@U�A�rC�kA<�AʵF@:weA��Af�A��@��@��uAoP�A|C?A�_R@���AmbPA���B�C��
A�> �
A���@�A�v�A��A��A4��A>UAWh�AF�Al�E@� �A��B;�A��A���@��?���A�tA��'A�\A_MAo�A��A�p�A*�A�r�A�~�A�|!A��LAW%xAӁ~A<y@T��A��C��A@��A�~�@,�A$�AHyA��@VS@���Ar�KAx�]A�yG@��mAm߉AÈ+B�rC��A�Y   (   
               .                     _      &      (      2      $      +      $   !   
   	   `     8      $      I          *      '      +         "                  {      %                  1         !            O                  )      %      %      %            -      -            %   %      #      $   +         !   )                        !                                       9                  '            %      !            #                     !      !         !         !   )                     Oʝ�N��N�YO�N`.�O�S�O�0�N�	O� �O!�N�D(O]��O���P�ޮN���N��PO�,O���NQo�P%�N�*EO��N�`O��Ny�O�1xOD:�NCQ�N�2aO��nO[6O���N��N�N�I�O���O�ȼN|ÈO��;N0�OpJ�O�>O��N-��O��O�ANB�*N$�N�TO7�N�hO*�NIM    �  �  X  y  ^  -  �  �  �  �  X  %  o  �  �    �  �  9    �  �  �  	  L  �  �  �  
  2  �    X  �  �  >  �  T  �  G  �  �  �  [  �  �  �  �  '    |  ⻃o�ě���o<o��o:�o<ě�<o<D��<e`B<D��<T��<e`B=@�<�C�<�=\)<���<�/=+<�h=�w<�=o=\)=�w=#�
=\)=�P=��=#�
>333=0 �=m�h=8Q�=u=L��=<j=]/=H�9=}�=aG�=aG�=ix�=m�h=m�h=q��=�%=�+=�O�=��P=�x�>   ��������������������
#./850/#

��������������������-*/0<GHIUZ__ZUH<9/--���������������

"/19AFHKH;/"b_]cfhlt���������thb)5>85)��������������������Y[gt�����������{tg_Y##+/<@GC=</.$#######-<E<<OSXfijgjfbUI<0-������������������������5@JRV`d[NB)���
 )/33.)-'#*/<HRUWYUOH</----��������������������UW]gt�����������rl^U��������������������fa^]gt������������tf�����
 #''#
�������� "%((% ��##�������%�������)/0))!�������������������������������������������������������������������������������������)5>?7)������������

�����@AEN[gt������tg[NGB@��������������������&!))6BCOOSOOIB66-)&&�������������������� ��
"*<@C?</#
#/;=>AEMH</����������������������������������������*/1+*'����������������������������������������������'-2.0)���Z[ahrt{trha[ZZZZZZZZ��)5=@NUZNB5)|{����������������|��������������������('"6CFIC@6**66666���������	

���

 









�������	 ��������)15765)'���ùϹܹ������蹾�������������������A�B�N�N�W�N�M�A�5�(�'�"�(�(�5�9�A�A�A�A�������������������������������������������������������������������������������������������������������������������������	��"�;�H�V�W�M�H�;�/�"��	����������	����(�A�F�M�X�T�M�A�4�(����������A�M�Z�b�e�f�Z�M�H�C�A�@�A�A�A�A�A�A�A�A�׾���	���$�&�"��	�����׾ʾž��ž׾���������������������|�s�o�l�s�u�z����y�y���������y�m�`�`�`�`�m�y�y�y�y�y�y�y�4�<�M�f�����������r�f�Y�M�;�*�(�.�1�4�T�a�h�m�m�g�_�T�F�;�/�"����"�/�;�H�TƧ��������������ƧƁ�\�C�1�*�,�4�WƁƎƧ¦±²´²­¦�z�x������������
��������������������뻷�ûǻлڻۻлû������������������������3�@�L�Y�c�f�`�Y�@�3�'��������'�3�a�h�n�q�n�n�g�a�[�V�U�S�U�]�a�a�a�a�a�a�����	��+�6�<�<�5�/�+�"�	���������������;�D�H�T�W�Y�T�Q�H�;�7�/�"��"�%�.�/�0�;�	��;�G�W�`�^�P�G�;�.�"��	���������	�������������y�y�y�{���������������������)�B�]�k�h�_�R�N�C�B�5������
���)�5�A�N�P�Z�Z�]�Z�N�B�A�5�1�2�5�5�5�5�5�5���Ľݽ��������нĽ���������������ù������������������������ùðïììôù�<�H�T�Q�H�<�/�/�/�0�<�<�<�<�<�<�<�<�<�<�a�c�m�r�z�m�a�T�T�H�A�H�T�W�a�a�a�a�a�a�����������������������������������������׾����	�����	����۾׾ʾ����¾ʾ�����)�0�3�1�)���������������������A�B�M�Y�Z�a�\�Z�M�A�9�4�2�4�4�;�A�A�A�A����������������ۺ������������������������������������E�E�E�E�E�E�E�E�E�E�E�E�EiE\EUEREZEiExE��4�M�Z�f��������s�f�M�4�(������(�4ÓÝàêåäàÓÊÊÊËÓÓÓÓÓÓÓÓ�����ɺֺ���	�
����ֺɺ����������������y���������y�l�l�l�q�y�y�y�y�y�y�y�y�y�y�������������������������y�s�m�o�t�y�~���g�������������������������s�N�A�8�A�N�g�r�~�����������������~�e�L�@�L�R�S�V�f�r�r�w�����~�r�g�f�^�f�n�r�r�r�r�r�r�r�r���������������������|�y�m�Z�R�M�R�`�m���������Ŀ�����(�*�����ѿ������������������������������������������������������������ ������������������y�o�m�i�g�i�m�y�������������������#�/�<�C�H�U�a�a�e�a�U�T�H�<�/�*�#���#ǺǭǧǡǔǒǑǔǡǭǺǺǺǺǺǺǺǺǺǺD�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D��������
���������
�
������� , - / $ 1 4 . A 5 8  ^ C I    8 & b > U ; E 8 H N  . P & 3  8 - ] ` ` W > / 1 a - 0 C m  6 D 3 f , �    �  �      j  #  �  �  �    �    �  �        ?  �  �  C  A  @  �  �    �  \  �  (  �  `  �  �  �  �  �  �  �  ,  �  K  �  E  �  �  M  3  �  >  \  )  �  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  �  �    �  �  �  �  �  �  �  |  Q    �  o  �  y  �  )  >  �  �  z  i  X  G  7  &      �  �  �  �  m  0  �  �  �  o  R  j  }  �  �  �  �  }  x  t  l  d  X  <    �  9  �    ~  �  �      &  )  I  X  V  I  2    �  �  �  5  �  ]  �  Y  y  u  q  n  j  f  b  _  [  W  Q  H  ?  6  -  $      	     ^  [  [  Y  W  R  L  B  6  %    �  �  �  T  �  �  C  �  F  +  �  �  �  �         (  ,  #        �  �  V  �  �   `  �  �  �  �  {  u  p  h  `  X  O  F  <  2  '        �  �  z  �  �  �  �  �  �  �  �  ~  m  W  ?  %    �  �  �  :   �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  j  D    �  �   �  �  �  �  �  �  �  �  �  �  �  l  Q  .    �  �  @  �  �  9  X  I  :  /  %           �  �  �  �  �  �  j  D       �  %      
  �  �  �  �  �  �  �  �  �  �  �  j  D    
   �    9  5  2  E  b  n  k  Q  ,    �  �  o    �  �  �  �  C  �  �  �  �  �  �  �  �  �  �  ~  j  T  <    �  �  �  �  c    C  k  �  �  �  �  �  w  U  +  �  �  y  -  �  �  &  �  K  �  �  �  �  �  �  �  �      �  �  �  c  #  �  �  n  -  �  �  �  �  �  �  w  O  ,      �  �  �  I  �  h  �  �  �  �  �  �  z  \  6    �  �  �  _  1    �  �  r  B    �  �  �  ,  8  6  '    �  �  �  �  �  }  V  4    �  �  L  �  �  ^    �  �  �  �  �  �  �  v  b  N  :  -      �  �  �  h      T  s  �  �  �  �  �  �  y  d  H  %  �  �  b  	  �  A  �  �  �  �  �  ~  s  h  ]  N  ;  (    �  �  �  �  |  I     �  �  �  �  �  Z  ,  �  �  �  _  &  �  �  B  �  T  �  �  �    �  �      	  	    �  �  �  �  �  h  ,  �  �  y  S  3    2  :  F  K  K  ?  '  	  �  �  �  N    �  �  h    �  �  3  �  �  �  �  �  �  �  �  y  M    �  �  R  �  �  B  �  u  �  �  t  `  K  7  "    �  �  �  �  �  t  U  4    �  �  F  �  �  �  �  z  k  \  K  :    �  �  �  �  g  A    �  �  �  w  	  	P  	�  	�  
  
  
  
  	�  	�  	1  �  S  �  �    s  �  �  %  *  0  /       �  �  �  �  a  +  �  �  W    �  �  y  M    �  L  ,  �  +  (  �    �  �  S  �    
  �  �  q  �  �  �      �  �  �  �  �  �  �  �  w  ^  D  &    �  �  �  }  X  �  �       1  C  R  X  W  L  3    �  �  Q    �  {      �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  H  w  �  �  �  n  B    �  ]  
�  
T  	�  �  �  �    �  H      7  >  >  7  *    �  �  �  y  x  R  '    �  �  �  �  �  �  �  k  H  %    �  �  �  l  B    �  �  �  �  �  �  �  &  C  R  T  I  1    �  �  �  K  
  �  �  N  
  �    %   �  �  �  }  v  o  h  a  X  N  D  :  0  &         �   �   �   �  }  �  �    6  C  G  C  9  &  	  �  �  �  9  �  U  �  N    �  �  �  �  �  �  �  �  �  �  }  `  9    �  �  t  A    �  �  �  �  �  �  �  l  ;    �  �  B  �  �  ^    �    A  p  �  �  �  �  �  �  �  �  r  b  Q  @  .    �  �  �    E    [  U  D  *  
  �  �  �  �  U    �  �  N    �  �  B  �  5  �  �  ~  f  J  -    �  �  �  �  j  @    �  M  �  �  �  �  �  �  �  �  �  �  z  _  D  #    �  �  �  l  A     �   �   �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  s  _  L  9  �  �  �  �  �  v  `  K  5      �  �  �  �  �  `  <    �  '    �  �  �  �  �  e  <    �  �  r  ;    �  �  =  �  3          �  �  �  �  �  �  �  �  �  {  [  9    �  �  �  �  #  �  9  e  x  y  e  A  �  �  H  �    �  �  	      �  �  �  �  �  �  �  �  �  �  �  {  i  X  ;  	  �  �  !  �  0