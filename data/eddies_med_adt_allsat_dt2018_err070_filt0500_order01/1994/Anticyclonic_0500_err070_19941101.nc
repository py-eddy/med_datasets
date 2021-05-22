CDF       
      obs    3   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type                     	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?`bM���   max       ?��Q�      �  �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       M�AI   max       P�,�      �  x   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       ��`B   max       >C�      �  D   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>�33333   max       @E��\)     �      effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ��=p��
    max       @vZ�Q�     �  (   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @%         max       @Q�           h  0    effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @�R        max       @���          �  0h   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       ;�o   max       >��y      �  14   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A���   max       B+��      �  2    latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A��f   max       B+@      �  2�   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       =Ǒ�   max       C���      �  3�   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       >?X�   max       C���      �  4d   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max         K      �  50   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          C      �  5�   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          -      �  6�   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       M�AI   max       PH�      �  7�   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?��#��w�   max       ?�͞��&      �  8`   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       ��o   max       >C�      �  9,   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>��G�{   max       @E��Q�     �  9�   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ���G�|    max       @vZ�Q�     �  A�   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @#         max       @Q�           h  I�   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @�R        max       @��          �  JP   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         ?�   max         ?�      �  K   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�Fs����   max       ?���l�D     �  K�   G  J      D                     8                        T   0               H                  (      	         D         �   '            	   /                  P�,�P�5�N�@OԱ�N��O���O��OA��ND N��fP# �OT��N��cP�O+�GO�N�_N�%�P���PO��O9AN�d�O+��Oa��O��N�L�O���OއO s�O��O��{N`C�NF
(N�uOû&O���O�QXP-��Oq�OM���N�)O*T�N�g<OѤ�M�AINZ�>N�>N-~VM�yDNܣ9��`B��o��o:�o;o;D��<t�<#�
<49X<49X<D��<D��<T��<T��<T��<e`B<e`B<u<���<�j<�/<�`B<��=C�=\)=\)=�P=49X=49X=49X=8Q�=8Q�=<j=@�=H�9=Y�=ix�=ix�=q��=q��=�%=�o=��=�+=�7L=��w=��w=��T=��T>$�>C�PMXa���������������P""+5Nt���������gNB0"������

������������������������������������7;F[t��������tmj[NB7~�������������������������������������CJDB962156ABCCCCCCCC��������������������5BNmw}|tg[NVTW[hlpt������tohb[V������������������������#/7GSU^WJ/#
���gnoz������������zpng))*-/<HIRUadaUHD<2/)ZSSY[bhntqh[ZZZZZZZZ������������������������5BN\beb[N5)	��!" )0O[htz��}th[OB3!��������������������	��	"/2:7/#"					������������������������������������������������

����������������������������� �����������(5FNW[``eb`[P<0)������

������
#/232/-$#
����)42-% �������������������������#0<A<:0,#B?ABOXZROBBBBBBBBBBBXTV[ghkhg[XXXXXXXXXX��������������������34<GUan��������zaUJ3�������)����������6OZckoo\TB)��LKJJGO[cht}��~tkh[OL����������������������������������������517;BHKTaffca_[VTH;5��������������������adqx������������zma,.158BGGGB=5,,,,,,,,ntz��������zzqnmlnnn��������������������/.+-/<D@<0//////////���������������������������������������������A�(�����5����6�O�[�o�v�r�[�O��������������������)�6�B�B�O�S�O�L�B�>�;�6�)�)�)�)�)�)�)�)�4�A�Z�u�}�s�Z�M�A�(������������4�!�)�-�:�>�:�-�+�!��������!�!�!�!�m�y�������������y�m�`�T�G�C�G�T�\�U�f�m��������'����Ϲɹ��������Ϲ�軑�������ûлԻлɻ����������������������_�S�G�E�:�1�:�G�S�W�`�a�_�_�_�_�_�_�_�_����������������������������������������²¿������������²¦�t�c�X�O�U���������Ŀѿݿ�ܿѿ��������|�y�r�y�������������¿���������������������������������=�N�a�a�Z�A�(��������������$�(�4�A�E�N�R�W�P�N�A�5�$���	��������������� �����������������������(�/�(�%��������������ù��þ��������ùöìàÞÙÜßàì÷ùùƁƎƧ��������������u�c�X�L�O�h�|Ɓ���������ʾ�����ʾ����}�`�Q�Y�g�s���'�4�<�@�A�;�4�'���������������$�'�T�a�m�z�|�����~�z�m�a�]�T�T�Q�T�T�T�T�T�����ʼּۼּӼʼȼ����������������������������������������������������������DoD{D�D�D�D�D�D�D�D�D�D�D�D{DqDoDeDbDaDo�/�<�H�a�s�w�n�a�U�H�<�/�)�#� �'�&�&�*�/�"�.�;�G�I�M�G�;�.�"���"�"�"�"�"�"�"�"�ݿ���(�5�>�A�O�N�A�5������ݿѿÿĿݿ"�.�;�B�G�J�G�;�.�)�"��	������	���"�T�`�m�y���������}�y�m�`�X�T�G�F�?�G�O�T�~�������������������e�Y�@�1�+�0�@�L�e�~���������Ľνսнʽ��������y�l�`�l�{�����ּ�����������ؼּӼֺּּּּּּּ����������ܺں������������������������������������������������������������۹ù��������x�w���������������	�������	������������������������������������������������������������'�B�O�_�q�f�M�@�4�'�����߻׻ڻ��������ʼּڼּܼҼʼ�����������{�t�����M�Z�]�f�s�z�s�f�Z�W�M�M�M�M�M�M�M�M�M�M�������r�f�Y�W�Y�f�r�t���������I�U�b�n�{łŇŐœŇ�{�n�e�b�U�I�F�@�E�I��(�4�5�6�4�1�(���������������������&�/�2�)����������������������0�<�<�<�4�0�0�0�#�#�#�,�0�0�0�0�0�0�0�0�0�<�I�R�U�Z�U�I�<�3�0�+�0�0�0�0�0�0�0�0��� �!�������ܻػܻܻ�����������
�����
������������������������ĳĳĿ��������ĿĺĳĳĳĳĳĳĳĳĳĳĳE+E7ECENEPE\E`EgEbE\EPECEAE7E5E*E*E+E+E+ H  � D = = 3 + b Q G Q 2 3 b < < P 8 , 1 8 ; M . 7 7 i H  B 4 D 6 8 b j P ? 6 � p [ : H j R @ j G B    �  e  �  �  �  i  i  �  N  �    �  �  �  �  >  }  �  �  �  W  0  �  �  �    �  %  O  S  ,  k  �  T  +  #  $  >    �  X  [  �  �  �  "  �    {  %  =�o>��y;�o=�C�;�`B<���=t�<�<ě�<�9X=�7L=��<�1=49X<�h=�P<�1<�/=��`=�hs=L��=,1=��=D��=�"�=u=49X=y�#=m�h=�o=�1=��=]/=T��=P�`=���=�\)=�C�>@�=ȴ9=�C�=�O�=��w=��P=�l�=��
=�{=��=�9X>V> ĜB פB	J�B"�B"��B�B	CB��B"�5B�B��B�B^B)_B�-B�B�B��B!��B�B�B!@A���B"�vBX�B�B�B�AB�#B%Bc�B�B+��B%��B�B�nBw'B@B�B��BvKBB�A���B�IBCB�xB`B�2B�`B�/B��B>�B	?�BFdB"B��B	CkB��B"��B=�B�eB:�B��B?�BĜBCfB �BB�B!�hB�B=JB!GA��fB"�BE�B��B��BFB�)BFoBC�B��B+@B%oB��B	�B@SB�B��B=�B��B:�B�OA��
B�[B@=B��B�B��B;BEJB�1A�B�A�~4A׳>A9BQ@i�3Am4M?'�]@�S�A՘AС�A�̲As[�Au,�A�j�A���A�kA3�uA�֫B��AK4q@���A��@��A���C��OA�5UAa��A��"A_;�Aj6"?�Y�A !uA�9@M�A�q�=Ǒ�A���A�B�@�R(@�A@�@��A�9A5��A�F�A�3A��@�}�A�3
A⫝̸C���A���Aԉ�A؋SA9ٴ@l�Am
�?G�2@���A�XAЍ�A�ɾAs �AuA�&/A�y
A҃JA2�zA�~6B�AJ�G@�A���@�A�92C��zA�t�Aa��A��hA_�
AjG?���A"8�A �@SAA��>?X�A�{�A��K@��@���AAќ@�HA�z�A5Y�AӪ�A�kA�U@�'	A���A��C���   H  K      D                     8      	                   T   1               I                  (      	         E         �   (            	   /               	      C   3      %         %            '         )               3   )                        %         %   !            '   #   !   -                  !                     -   !                           !         )               !                                    %                  #      #                                    PH�P/`N�@OYp8N9jYNe�}O]�OA��ND N��fO���N���NYP��O+�GNrN�_N���O�)�Ok�FO��N�@�N�d�Nxd�OV��N���N�L�O�8XO�+O	��O��O��eN`C�NF
(N�uO��9O���OGOO�x�Oq�OM���N�)O*T�N�g<O�aXM�AINZ�>N�>N-~VM�yDNܣ9  k  s  �  ;  3    `  `  Q  ;  �  �  c  �  �  �  �  O  �            \  -    �  �  �  #  d  �  �  1  �  �  Y  �  F  �  �  �    :  .  e  �  �  ;  	<u>$ݺ�o<�j;D��<�o<���<#�
<49X<49X<�/<�1<u<�o<T��<�j<e`B<�C�=q��=49X<�/<�<��='�=�P=49X=�P=8Q�=8Q�=<j=8Q�=H�9=<j=@�=H�9=�o=ix�=q��=��=q��=�%=�o=��=�+=��P=��w=��w=��T=��T>$�>C�hcdm��������������yh?:>EN[gt�������tg[N?������

�����������������������������������������ZT[]gtyztng[ZZZZZZZZ����������������������������������������CJDB962156ABCCCCCCCC��������������������)5BN[hlljg[NB:Y[^ght���{tohd\[YYYY������������������������
#/DMURH/#
���gnoz������������zpng./2<HHTMH<1/........ZSSY[bhntqh[ZZZZZZZZ��������������������)5BGLNONMEB5)"?;;<?BJO[hmstqmh[OF?��������������������	"/73/"	������������������������������������������������

�������������������������������� �����������%)5BHN[bb_[N:.)�������
�����	 
#/121/+#
		����)42-% �������������������������#0<A<:0,#B?ABOXZROBBBBBBBBBBBXTV[ghkhg[XXXXXXXXXX��������������������34<GUan��������zaUJ3������%������ )6BO[ab]OB6 LKJJGO[cht}��~tkh[OL����������������������������������������517;BHKTaffca_[VTH;5��������������������z����������������zz,.158BGGGB=5,,,,,,,,ntz��������zzqnmlnnn��������������������/.+-/<D@<0//////////�������������������������������������������s�N�5�(��(�>�Z�����6�B�N�V�Y�W�R�E�6�)���������������)�6�B�B�O�S�O�L�B�>�;�6�)�)�)�)�)�)�)�)�(�4�A�M�Z�b�h�f�]�M�A�4�(�������(�!�"�-�7�-�'�!������� �!�!�!�!�!�!�m�y�����������y�n�m�k�m�m�m�m�m�m�m�m�m���������	�����ֹܹϹɹϹѹܹ�軑�������ûлԻлɻ����������������������_�S�G�E�:�1�:�G�S�W�`�a�_�_�_�_�_�_�_�_����������������������������������������²����������º²¦�{�t�l�n�t²������������������������������������������������������������������������������������8�E�N�Z�]�A�(���������������$�(�4�A�E�N�R�W�P�N�A�5�$���	�������������������������������������������(�/�(�%��������������àìù����ùòìãàÜßààààààààƚƧ������������������ƚƎ�u�p�i�n�uƁƚ�����������¾ʾϾ׾Ӿʾ����������u�x����'�4�<�@�A�;�4�'���������������$�'�a�m�v�y�y�x�m�a�_�V�T�R�T�W�a�a�a�a�a�a�����ʼּۼּӼʼȼ�������������������������������������������������������������DoD{D�D�D�D�D�D�D�D�D�D�D�D{DrDoDfDbDjDo�H�N�U�a�i�n�p�n�g�a�U�H�<�;�4�8�<�D�H�H�"�.�;�G�I�M�G�;�.�"���"�"�"�"�"�"�"�"�ݿ�����'�1�5�;�?�5�(�����ݿѿƿͿݿ"�.�;�@�G�I�G�;�.�'�"��	���� �	���"�T�`�m�y�������|�y�m�`�\�T�I�G�F�G�S�T�T�~�������������������e�Y�@�1�+�0�@�L�e�~���������ƽνĽ��������y�l�g�l�o�~�������ּ�����������ؼּӼֺּּּּּּּ����������ܺں���������������������������������������������������Ϲܹ������ܹԹ������������������������	�������	�����������������������������������
� �����������������������׼��'�6�D�M�R�S�M�@�4�'�����������������ʼּڼּܼҼʼ�����������{�t�����M�Z�]�f�s�z�s�f�Z�W�M�M�M�M�M�M�M�M�M�M�������r�f�Y�W�Y�f�r�t���������I�U�b�n�{łŇŐœŇ�{�n�e�b�U�I�F�@�E�I��(�4�5�6�4�1�(����������������� �+�/�)�#�����������������������0�<�<�<�4�0�0�0�#�#�#�,�0�0�0�0�0�0�0�0�0�<�I�R�U�Z�U�I�<�3�0�+�0�0�0�0�0�0�0�0��� �!�������ܻػܻܻ�����������
�����
������������������������ĳĳĿ��������ĿĺĳĳĳĳĳĳĳĳĳĳĳE+E7ECENEPE\E`EgEbE\EPECEAE7E5E*E*E+E+E+ T  � 5 G @ + + b Q > 0 , 1 b + < 0 B + 1 E ; 2 *  7 ^ I   B 6 D 6 8 Y j ) 6 6 � p [ : " j R @ j G B    �  .  �  �  M  y  F  �  N  �  �  �  i  F  �  w  }  �    �  W  �  �  �  �    �  �  =  -  ,  "  �  T  +  �  $  �  �  �  X  [  �  �  L  "  �    {  %    ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  �  L  �  �  $  ]  j  @      
  �  �  �  �  i  �  '  !  �  �  p  T  �  �  �  ,  h  r  X    �  �  �  �  �  t  o  
�  �  �  �  �  �  �  �  �  `  E  8  *      �  �  �  �  �  �  y  �  H  �  �    0  ;  8  -    �  �    :  �  r  �  �  �  �  .  /  1  2  1  $         �  �  �  �  �  �  �  �  �  �  �  �  �  �    '  y  �  �  �  �  
      �  �  �  �  @  �  T  �  �  
    1  C  ?  ^  Y  I  ,    �  �  �  �  X    �  �  `  `  [  L  1    �  �  �  �  �  �  �  o  V  #  �  }  &  �  Q  5    �  �  �  \  J  B  J  x  g  B    �  �  �  w  K    ;  8  5  .  &        #  )    
  �  �  �  �    V  +  �  �  �  �  �  �  �  �  �  l  H  8  0  +    �  �  )  �  �  �  d  W  O  `    �  �  �  �  �  n  J  #  �  �  �  C  �  �  ^  V  [  _  b  b  b  _  [  P  C  2      �  �  �  �  a  :    �  �  �  �  �  �  �  �  |  g  P  5      �  �  ~  5  �  #  �  �  �  �  r  V  8    �  �  �  �  �  X  (    �    w   �  �  �  0  x  �  �  �  �  �  �  �  �  j  A    �  �  a  �  J  �    x  o  c  W  J  >  1  "      �  �  �  �  �  �    b  9  2  6  K  @  ,    �  �  �  9  �  u  B    �  �  �  6   �    �    N  e  t    �  �  �  �  v  6  �  {  �  ?  v  �  ^  ?  k  �  �  �  �  �          �  �  �  X    �  �  U  [        �  �  �  �  �  �  �  �  �  �  �  }  3  �    S  �  �  �  	    
    �  �  �  �  �  �  �  z  d  I  �  �  I  �                �  �  �  �  �  �  �  �  �  w  ]  @  $  K  Q  ]  l  s  r  r  t  |    x  h  O  3    �  �  �  ~  :  W  L  I  U  U  B     �  �  <  �  =  �    �  
�  
/  	  �  �  �  �  �  �        +      �  �  �  A  �  �  :  �  o  �    �  �  �  �  �  �  �  �  �  �  p  \  I  5  "        �  �  �  �  �  �  �  r  Z  D  C  ,    �  �  �  N  
  �  h    �  �  �  {  r  f  U  ?  %    �  �  �  G    �  t    �  	  �  �  �  �  �  �  �  �  �  r  X  8    �  �  �  T    �  	  #    �  �  �  �  �  f  B  X  4  
  �  |    �  =  �  �  �  W  _  c  c  ^  S  C  /    �  �  �  d    �  c  �  X  �   �  �  �  �  �  {  u  i  ^  N  =  ,      �  �  �  �  y  >    �  �  �  �  �  �  �  �  �  �  {  q  v  �  �  �  �  �  �  �  1  /  .  ,  *  )  '  %  #  "      �  �  �  �  �  �  �  v  7  y  �  �  �  s  O    
�  
i  
  	�  	4  �  3  j  �  �  �   �  �  �  �  �  �  �  �  q  [  D       �  �  �  �  �  �  �  �  =  M  X  Y  V  Q  M  J  D  ;  +    �  �  �  �  [  *     �  R  �  P  �  �  �  �  �  �  @  �  f  �  S  
�  	�  �  |  �    F  C  A  8  (  
  �  �  �  �  �  W  $  �  ~    �    �    �  �  �  �  v  d  R  @  (    �  �  �  w  R  .    �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  v  f  X  J  ;  �  �  �  �  �  a  5    �  �  �  k  Z  ;    �  �  �  2   �        �  �  �  �  �  �  �  �  �  m  S  :    �  �  W    �  �  �  9     �  �  y  5  �  �  L  �  �    �  N  �  h  �  .  #         �  �  �  �  �  �  �  �  �  �  �  �  �  �  ~  e  V  G  7  &      �  �  �  �  �  �  �  �  �          �  �  �  _  2  �  �  �  �  T  '  �  �  �  h  9    �  �  �  �  �  �  �  n  Z  E  0      �  �  �  �  �  �  �  �  o  P  ;  (        	          	    !        �  �  �  �  	  �  �  q  +  �  �  Y    �  P  �  �  .  �  k      �  �