CDF       
      obs    :   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type                     	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?`bM���   max       ?щ7KƧ�      �  �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       M�~�   max       Pѯ�      �  �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       ����   max       =���      �  |   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>��
=p�   max       @E�
=p��     	   d   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?������    max       @v�G�z�     	  )t   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @$         max       @R�           t  2�   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @̝        max       @��          �  2�   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       �o   max       >�E�      �  3�   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A�ʊ   max       B,;�      �  4�   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A�{Q   max       B,D      �  5�   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       ?�6O   max       C���      �  6�   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       ?���   max       C���      �  7�   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max               �  8h   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          I      �  9P   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          1      �  :8   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       M�~�   max       Pi�      �  ;    speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�dZ�2   max       ?���o      �  <   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       ����   max       >6E�      �  <�   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>��
=p�   max       @E�z�G�     	  =�   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?������    max       @v�33334     	  F�   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @         max       @R�           t  O�   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @̝        max       @��          �  Pl   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         G   max         G      �  QT   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?���,<�   max       ?��Mj        R<                  
   )   $   !            J      y   <               a      *      P            	   
                     0         8         &   =      
               4   s   E              NB�CN0b�O���N�OtF_O��O�?O��(O�iN;�5N EO���PK�N��P�O�jgO��N���OţN�lqPѯ�N���O�H{N�	�P%��M��O'g�N�"bN��)N��N�O[�O�N��OM	�O���PS�N��N��bP*��O>ҼO���OvOe@�O��O%
O�M�~�N�LN��O�gOx�jP7AnN�9P6j�O&�NܬN��~����t��D��%   :�o;�o;ě�<t�<t�<49X<T��<T��<�t�<�t�<�9X<���<���<���<���<�/<�h=+=C�=C�=\)=\)=t�=�P=��='�='�='�=49X=49X=49X=<j=<j=@�=L��=T��=T��=Y�=]/=e`B=m�h=m�h=u=y�#=y�#=�o=�hs=���=��T=���=���=�1=���=���$')6BBKB964)$$$$$$$$��������������������??DUanz��������zaUH?�������

�����"$/;BHJHHC;/)"!���������#<HUanz}ztnU</#����)6<;>;6-)��������&+,#
������� $%��������������������������������������������������������$)6A@:64+)&_XZan������������zn_xutrz��������������x��������������������������������������������������������OLLO[hmtutnh[OOOOOOO�����6QZ^_ig[N<�����������������������dadlt������������}td���������������������������#*-(
��������������������������ehktx������������rheQJKU]anoz{�zna_UQQQQ������������������������	

 �����

�����������������������������),-)(��������������������� )/68?MND5)Qfqss�������������gQZU[`hlt����}vth[ZZZZCCOHBA6)(!)6CCCCCCC�����)0<><5)��������
#/<FH<</'#
�����������������������������������������������
-19=<6/#
��������������������������������������������snqyz�������������zsA@BHNOPQSOIBAAAAAAAAPSY[hskh_[PPPPPPPPPP��������������������}����������������xx}�����
��������������()#�������/# "#%'/<HF=<////////-4BNgt�������g[NB8/���������

�������������������������������������������ÇÓÔÙÓÓÇ�|�z�w�zÀÇÇÇÇÇÇÇÇ���������������������������������������������������������������������������������A�F�N�Z�g�s�y�s�p�g�Z�N�A�5�2�0�5�>�A�A�T�a�m�y�z�����z�m�a�H�;�/�$�%�/�8�;�M�T�/�<�H�U�\�a�f�i�a�Z�U�H�G�<�/�*�*�)�/�/�������������������������������뼘�����ʼҼӼռѼʼ��������r�_�^�j�r���������û˻Ȼ��������x�X�I�I�K�S�l�x���������!�-�2�-�&�!���������������������������������������!�-�:�D�7�D�@�:�-�!������ٺ���лܻ���������ֻܻû����������������п������������������������y�x�y����������F=FVFoF�F�F�FFsFcFJF=F$FFE�E�E�FFF=ù����������������������ùêÞÖ×Üàùàìùü������ÿùìàÓÇÃÄÇÓÕÓà�zÄËÓàê×ÓÍÇ�z�tÇÊÇ�n�i�m�m�z�M�Z�f�j�}�����s�f�Z�A�3�(��(�0�A�J�M����������� �������������������Ƴ����8�=�����Ƴƚ�u�O�3�$�(�C�[�^�kƳ���"�/�9�/�+�"��	���������	�������������%�.�4�2�+�"���������������������'�-�0�1�'�����������������B�N�n�t�~�t�n�[�N�B�5������������)�B���ʾ׾پ޾׾ʾ��������������������������/�3�<�B�H�S�^�\�U�H�<�/�)�#����#�,�/�����������������������������������������
�������
������������
�
�
�
�
�
��"�.�8�;�;�;�2�.�"���	��	�	�����`�m�o�m�j�`�T�G�A�G�T�\�`�`�`�`�`�`�`�`�!�-�:�G�S�_�l�r�x�|�x�l�_�F�-�!����!�;�G�Q�R�G�B�;�.�"��	���	���"�.�:�;���ʾ׾��������׾̾ʾ��������������Ľнݽ����ݽнĽ������������������Ŀ����������������������y�m�`�U�T�`�m�y���y�����ѿ��(�%�����濸�������z�o�q�p�y�3�@�E�L�U�Y�^�Y�S�L�J�@�3�2�.�3�3�3�3�3�4�3�'����
����'�-�,�4�4�4�4�4�4�4�������������������������������������Ž������!�#�$�!���������������������ĽŽ½������������y�r�l�b�l�w�����@�M�Y�i�r�����}�r�f�Y�M�@�>�3�0�4�9�@E�E�E�E�E�E�E�E�EuEiE\EQE\E`ElEkEqEuE�E��N�Z�g�u�{�s�Z�N�E�>�(��������"�A�N���������������������������������|�����������ʼּ׼ݼݼּ̼ʼ��������������������������������������������'�4�5�6�4�'�%������������¿������������¿²«²µ¿¿¿¿¿¿¿¿������	�����ݽͽĽ����������ݽ�D�D�D�D�D�D�D�D�D�D�D{DrDoDmDoDtD{D�D�D��������0�C�R�P�H�0�������ĿİĤĠĦĳ��ĳıĳĿ��������������Ŀľĳĳĳĳĳĳĳ����)�7�<�>�:�)�������������ÿ���������#�/�<�H�H�U�a�c�g�a�]�U�H�<�/�-�#���#�����ɺɺɺ������������������������������������������������~�z�r�p�r�x�~�������� I l b Z 9 * a / H J N , # F 9  . � M 6 V i  &  l 8 j T 0 n e F B : B J I E & h   K o l B ) o f C  - \ M 4 4 s O  i  j  z    �  ,  �  �  '  \  T  Q  M  �  p  �  =  �  �  �  \  �  8    �  %  �  �  �  �  6  �  Q  �  �  L  �  �  �  �  �    }  �  �  |  #  J  W  �  Q  �  �  �  7  K  I  �o��`B<#�
<#�
<���<e`B=<j=49X='�<u<�o=#�
=�j<�9X>n�=�{=H�9=�w=��=C�>%=�w=���=H�9=�=�P=u=,1=<j=L��=0 �=aG�=e`B=L��=�+=y�#=�v�=�%=}�=�/=�o=��w=�j=�h=��-=�7L=��w=��=�+=��
=��#>A�7>��=��>�E�=��>J>\)B��B"ŢB��Bm�A�ʊBnB�B��B#��B�HB�&B ɕB��B]�B�lBE�B!��B�B#V�Be^B��B}�B
�BB!F�B�-B�<BU(Be�B�IB�zB%rB+e�B��B�"B!WB��B5�B�!B�2BdEBu]B,;�B�B��Bd�B��B��Bz�B^�B%B�B��Bh�B3KB	$:B�B�1Bl�B�hB"HB�tBG�A�{QB�B�B�B#:�B�.B�_B �BH�B{;B�B@�B"<�B��B#=B�qB=?B>�B
��B![�B�pB��BkOB��Bb�B�&B1 B+7�B��B�B!@B��B��B9�B�^B��B�B,DB��B�IB�	B��B�`B�2BD+B?�B��B�B="B?�B	=�B6�BBB�Aɺ�@�p�A��6A���A���AĝNA�r�@��l@���@gt�A��@\�X@�w�Aq��C���A�tA��A�a4A>��A1(�B�A�s�A��j@�@A���AP�%A�GA��A�d�A^�XAg�@��A`o�AS AA&�uAn�FAw}1?��M?�6OA�h{A2{�ABw@�ôC�	�A�öA��@��P@�Ҭ@�|rA��A,o�C��!A��A��A�eA���@)V,@�AɃ@�7A��A���A�~�Ać�A�|�@�+/@��@d�mA�ZG@\i@���AqK�C���A�|�A�keAȀxA=�A0��B�cA��fA�ا@�y�A�^�AO+�A�|A��A��vA]`Ag�?@��A_vARɭA'�AkEAw�?ǻ�?���A��A2׶A(�@���C�dA�|�A��@��e@�D�@���A��A-�]C���A�A��AӃ�A�`@+�e@#b                     )   $   !            K      z   =               b      *      Q            	   
                     0         9         '   =      
               4   s   F                                         %            '      '                  I      '      )                                    1         )               %                        1      +                                    #                  !                  1      %      !                                    )         )               #                        -               NB�CN0b�O���NJ�2OtF_N�# N|h�O"Z�O��=N;�5N EO>�Oz�WN��O��LO��O��N���O�RN�lqPi�N���O��pN�	�O�ބM��N�n5N�"bN��)N��N�N�sO�N?��OM	�O���P�N��N��bP�O>ҼO0*�O+�Of-O��O́O�M�~�N�LN��O�gO'�@P*7�Nb�|O���O&�NܬN��~  5    �  �  y  �  �  �  E  ,  5  ?  |  p  �  	R  �  G  =    �  �  �  ,  �     �  K  t  �  "  �  K  &  �  J  �  �  �  J  _  �  	3    l  l  �  �  �  T  �  �  	b  �  �  �  �  ༛��t��D��;D��:�o;��
<�/<ě�<u<49X<T��<��
=<j<�t�=P�`<�<���<���<�/<�/=m�h=+=\)=C�=Y�=\)=49X=�P=��='�='�=0 �=49X=8Q�=49X=<j=]/=@�=L��=aG�=T��=u=m�h=�+=q��=q��=u=y�#=y�#=�o=�hs=��=�1=�{>6E�=�1=���=���$')6BBKB964)$$$$$$$$��������������������??DUanz��������zaUH?����


���������"$/;BHJHHC;/)"!���������,/2<HKUUUOH<1/,,,,,, �).//,)$ ��������&&#���� $%������������������������������������������������������������$)6A@:64+)&adnz����������zngdaazz}���������������{z������������������������������������������������� ������OLLO[hmtutnh[OOOOOOO������)6?FJA)������������������������dbelt������������~td����������������������������
! 
�������������������������}{������������}}}}}}QJKU]anoz{�zna_UQQQQ������������������������	

 �����

�����������������������������)*+)&��������������������� )/68?MND5)njqryy}������������nZU[`hlt����}vth[ZZZZCCOHBA6)(!)6CCCCCCC������).:<:3)������
#/<FH<</'#
�����������������������������������������������
")&#
��������������������������������������������snqyz�������������zsA@BHNOPQSOIBAAAAAAAAPSY[hskh_[PPPPPPPPPP��������������������}����������������xx}���������
 �������������&("
�����!#$&(/<=C<;/+#!!!!!!<::BN[gt������tg[NB<���������

�������������������������������������������ÇÓÔÙÓÓÇ�|�z�w�zÀÇÇÇÇÇÇÇÇ���������������������������������������������������������������������������������5�A�N�Z�Z�[�Z�N�A�6�5�2�5�5�5�5�5�5�5�5�T�a�m�y�z�����z�m�a�H�;�/�$�%�/�8�;�M�T�H�U�[�a�d�h�a�V�U�Q�H�<�/�-�-�/�<�=�H�H��� �������������������������������������������ļƼ�������������x������������������ûŻû����������x�j�_�V�R�^�x�����!�-�2�-�&�!��������������������������������������!�+�-�6�-�!��������������ûлܻ������ܻлû������������������ÿ������������������������y�x�y����������FcFoF|F}F{FuFgFVFJF=F1F#FFFF.F=FJFVFc����������������������ùìáÙÚàìù��àìùü������ÿùìàÓÇÃÄÇÓÕÓà�zÄËÓàê×ÓÍÇ�z�tÇÊÇ�n�i�m�m�z�A�M�Z�f�i�s�{�����s�f�Z�M�A�9�4�(�2�A����������� ������������������������������������ƳƚƁ�s�p�{Ƨ�������"�/�9�/�+�"��	���������	�������������$�-�2�0�*�"���������������������'�-�0�1�'�����������������)�5�B�N�]�f�g�d�Z�N�B�5�)����������)���ʾ׾پ޾׾ʾ��������������������������/�<�H�I�S�N�H�<�/�#�#�"�#�(�/�/�/�/�/�/�����������������������������������������
�������
������������
�
�
�
�
�
��"�.�8�;�;�;�2�.�"���	��	�	�����`�m�o�m�j�`�T�G�A�G�T�\�`�`�`�`�`�`�`�`�-�@�F�S�_�l�n�w�l�_�S�F�:�-�"�!��!�#�-�;�G�Q�R�G�B�;�.�"��	���	���"�.�:�;�ʾξ׾�����׾Ѿʾ��ȾʾʾʾʾʾʾʾʽĽнݽ����ݽнĽ������������������Ŀ����������������������y�m�`�U�T�`�m�y���y�������ѿݿ�������ۿ��������y�y�|�y�3�@�E�L�U�Y�^�Y�S�L�J�@�3�2�.�3�3�3�3�3�4�3�'����
����'�-�,�4�4�4�4�4�4�4�������������������������������������������!�#�$�!�����������������������������������������z�y�s�m�t�y���C�M�Y�f�r�����x�r�f�Y�M�@�@�5�3�4�?�CE�E�E�E�E�E�E�E�E�E�E�E�EuEtEoEnEvE�E�E��N�Z�g�r�q�Z�N�D�<�5�(�����	��#�A�N���������������������������������~�����������ʼּ׼ݼݼּ̼ʼ��������������������������������������������'�4�5�6�4�'�%������������¿������������¿²«²µ¿¿¿¿¿¿¿¿������	�����ݽͽĽ����������ݽ�D�D�D�D�D�D�D�D�D�D�D�D�D�D{DxDuD{D|D�D�ĳ�������0�@�L�P�N�F�0�������ĿĴĨĥĳĳĽĿ��������������ĿĴĳĲĳĳĳĳĳĳ�����!�+�/�/�,����������������������#�/�<�H�H�U�a�c�g�a�]�U�H�<�/�-�#���#�����ɺɺɺ������������������������������������������������~�z�r�p�r�x�~�������� I l b . 9 ( 3 ! I J N %  F /  . � Q 6 4 i  &  l  j T 0 n e F = : B 7 I E $ h # H = n B ) o f C  % \ ]  4 s O  i  j  z  i  �  �  �  V  �  \  T  �  �  �  w  _  =  �  D  �  �  �  $    �  %  �  �  �  �  6  /  Q  W  �  L  �  �  �  �  �  z  M  k  �  P  #  J  W  �  Q  g  i  �  �  K  I  �  G  G  G  G  G  G  G  G  G  G  G  G  G  G  G  G  G  G  G  G  G  G  G  G  G  G  G  G  G  G  G  G  G  G  G  G  G  G  G  G  G  G  G  G  G  G  G  G  G  G  G  G  G  G  G  G  G  G  5  ;  >  <  3  &    �  �  �  �  q  N  *    �  �  �  d  9    
  �  �  �  �  �  �  �  �  �  u  ]  E  ,     �   �   �   �  �  �  �  �  �  �  u  c  U  M  T  ?  !  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  {  _  <    �  �  I    �  y  n  X  <      �  �  �  �  �  �  f  1  �  E  �     N   �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  q  X  I  <  W  w  I  �  �  �    :  j  �  �  �  �  �  �  `    �     �  �  M  ~  �  �  �  �  �  �  �  �  �  �  �  �  �  S  �  �  @    C  +  .  4  @  E  <  /      �  �  �  �  �  �  V  +      �  ,  "        �  �  �  �  �  �  �  �  �  {  ~  �  �  �  �  5  /  *  %          �  �  �  �  �  �  �  �  d  G  *        .  6  <  >  ;  2  "  	  �  �  �  t  M  6    �  =  �  �  V  �  �  /  T  n  {  w  a  A    �  �  ;  �    S  }  7  p  l  h  d  `  [  T  M  F  ?  5  )        �  �  �  �  �  �  �  C  h  �  �  q  V  4    �  M  �  �  5  
M  	4  �  A  v  	  	M  	Q  	D  	,  		  �  �  e  &  �  �  P  �  �  	  r  �  �    �  �  �  z  n  S  8    �  �  �  �  �  s  f  b  p  k  Z  G  G  t  �  �  ;  J  -  	  �  �  q  1  �  �  d    �  �  ;   �  9  <  6  %    �  �  �  �  �  �  l  R  8    �  �  �  }  Q    w  n  f  \  S  K  E  ?  6  -  "    �  �  �  �  �  �  �  0  ?  F  �  �  �  �  �  �  �  �  R    �    X  g  >  �    �  �  �  �  |  f  P  9  !  
  �  �  �  �  �  �  `  &   �   �  �  �  �  �  �  �  �  m  H  !  �  �  �  �  y  J  
  �  v  \  ,  '  #            �  �  �  �  �  �  �  �  o  K  �  0  �  O  �  �  �  �  �  �  �  �  Q    �  R  �  F  �  �  �  [         �  �  �  �  �  �  �  v  B    �  �  q  =  	   �   �  �    S  s  �  �  �  �  �  v  X  3    �  �  m  .  �  �  �  K  I  H  F  C  :  1  (        �  �  �  �  �  �  �  �  �  t  l  e  Y  K  :  "    �  �  �  �  t  I    �  �  �  z  \  �  �  �  �  �  �  �  �  �  �  m  ]  U  J  <  (    �  �  �  "          
       �  �  �  �       	          !  �  �  �  �  �  �  �  �  p  Y  ?     �  �  �  J    �  s  ,  K  F  A  ;  2  )        �  �  �  �  ~  U  !  �  �  �  a  "  #  $  &      
  �  �  �  �  �  �  �  n  U  ;     �   �  �  �  �  �  �  r  T  4    �  �  �  n  +  �  �  8  �  �  7  J  3      �  �  �  �  |  \  :    �  �  �  c  <  $    5  �  �  �  �  �  �  �  �  �  �  \  -  �  �  w  7  �  �    Z  �  �  �  p  S  4    �  �  �  k  7  �  �  ~  3  E  i  /  �  �  �  �  �  �  m  X  D  2      �  �  �  �  �  �  �  �  l  ,  J  5    �  �  Z    �  p  *  �  �  h    �  f  �  �  �  _  M  ;  &    �  �  �  �    d  N  7      �  �  �  �  I  �  �  �  �  �  �  �  �  �  �  |  k  X  ?    �  �  +  �  -  	  	*  	3  	*  	  �  �  �    M    �  }  �  ^  �    M  �  X  �  #  �    �  �  �  m  *  �  i  
�  
q  	�  	^  �  �  �  |  o  ^  l  c  V  D  -    �  �  �  l  J  ^  i  3  �  �  �  O  �  i  k  l  e  ]  Q  D  3     
  �  �  �  �  �  b  E  (  
  �  �  �  s  H  !  �  �  �  �  �  f  ;    �  �  L  �  �  9  �  �  �  �  
    "  "  !        '  6  E  U  d  x  �  �  �  �  �  �  �  �  �  |  a  G  +    �  �  �  �  c  ?    �  �  �  T  .    �  �  �  �  z  ]  >    �  �  �  ]  �  u  �  R  �  �  �  �  y  o  e  U  >    �  �  {  3  �  i  �  )  ,  �  �  �  �  D  x  �  �  ^    �  *  �  �  S  �  }  �  �  	2  �  !  	  	a  	T  	1  	  	,  	  �  �  �  d     �    �  1  �  �  �  *  �  �  �  �  �  �  ]  /    �  �  �  e  7    �  �  d  3    �  s  �  u    �  �  �  �  K  �  �  �  �      �  "  �  �  �  �  i  O  $  �  �  �  f  5    �  �  �  G    �  r    �  �  �  �  l  U  >  '    �  �  �  �  c  <    �  �  �  j  ?  �  �  �  �  j  @    �  �  P    �    O  �  y  B    �  u