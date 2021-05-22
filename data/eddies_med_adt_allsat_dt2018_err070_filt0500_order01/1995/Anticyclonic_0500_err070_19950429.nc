CDF       
      obs    E   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type                     	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?`bM���   max       ?�$�/�       �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       N @�   max       P�C       �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       ����   max       =���       �   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>ٙ����   max       @F��\)     
�   �   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?���R    max       @vnfffff     
�  +�   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @.         max       @P@           �  6x   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @��        max       @���           7   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       ��9X   max       >A�7       8   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A�|>   max       B4��       9,   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A�u�   max       B4�`       :@   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       @�[   max       C�)z       ;T   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       @��   max       C�7d       <h   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          �       =|   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          ?       >�   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          9       ?�   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       N @�   max       P�q`       @�   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?���Y��}   max       ?��t�k       A�   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       ����   max       =��`       B�   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>ٙ����   max       @F��\)     
�  C�   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?��\)    max       @vl�\)     
�  N�   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @*         max       @P@           �  Y�   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @��        max       @�ـ           Z   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         @�   max         @�       [$   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�L�_��   max       ?��t�k     �  \8         	            c      )                  �      "            J   $   K      
      ;         A   S   %   O                        :   $      :   +   ]   =                     -   
   
                  v               /      	Nk�N���O�N���N��+N~��P�N���O��lO��CN�9XN��"NT��Nt�P`רO%�O��O�J�N�;P$�P6�O�7�P�CO�ЉNNZ	N6�P%�OCC�NDwPtuP��O�g�P<��O��N-��O]q�N�&�O���O���N�TOˉsO��N��P��O�*�P�P,�.NEsNe�CNm�N @�O�o!O�{$O��%N���N��N�PQO!�3N5�N{>�O��LP# N2BXNs(�O ��N�o�N�D�NxaxN-��������ͼ��
���
��t��o��`B��`B���
��o�D���o��o:�o:�o;�o;�o;��
;�`B<o<o<t�<#�
<49X<D��<T��<e`B<u<�o<�t�<���<���<��
<��
<�1<�j<�j<ě�<ě�<ě�<�`B<�<�<��=o=+=+=+=\)=�P=��=#�
='�='�=,1=@�=L��=L��=L��=]/=�t�=�t�=��-=�9X=�Q�=�v�=Ƨ�=Ƨ�=���! #0:710.#!!!!!!!!!!��������������������!#/6<GHLHHD<0/*#!'/0<HUVUNQH<:/''''''lnsz���������zxnllll����

�������436B[g�����vtqmg\B84dbcdfgoqttx|ytogdddd���
/<J\bd`UH/
����������������������#(0<>IIII?<40#��������������������NCOS[hlmh[SONNNNNNNN!#$��������,5<7)��������������������������������
"'((
���\mz���������������g\72:<=HLPH<7777777777�����/5HIB:5)�"/SUan|���nLD<3#FIJMPSX\ht������t[OF�������-4<)������"16;EHEJG;."�������������������������������������������)6BPSUUQB���Xdfchmz���������zmaX��������)5N[gt���~s[B0lkz����399(������zl�����������������������0LOMHB)����{������������������{���������������������
#/2=CIE?</#
snmrt}���������tssss��������������������TPR_g~���������{tg[T����������������������������������������522013;HTamoledaTH;5�����������������������)5N[ab^XMB5)���������������������������������������������5BGP[_\NG7)�;89:<@HRSMH<;;;;;;;;�
#$)&#
����������������������������DCHIUVZYUIDDDDDDDDDD�������������������#&'+*' ����������!*&#
����704<HUUZVUH<77777777srtu����������ztssssRT\ajmzzzmaUTRRRRRR�����������������������������������������{nhnn{��������������������������������������)6BF83)����25BN[_[TNB5522222222������ ������������JNO[[htx����{tsh[VOJ{|�����������{{{{{{������

��������#)+'#!���������������������ݽ����ݽнϽнٽݽݽݽݽݽݽݽݽݽݺ����������������������������������������������������������������������z�z�������T�_�a�m�h�b�a�[�T�H�?�F�H�N�T�T�T�T�T�T�����������������������������������������'�4�@�G�M�P�M�@�6�4�0�'� ��'�'�'�'�'�'�B�[�h�y��z�h�[�O���������������'�6�Bàìù����������üùìàÓÐÓÚàààà�`�m�z�������q�T�;�.�"������@�G�_�`����(�4�A�H�K�F�A�4������ݽٽ߽������������������������������x�u�v�x�}�������������������������������������������<�H�O�U�\�]�U�H�?�=�<�;�<�<�<�<�<�<�<�<��(�2�3�(������������������м�
�	�����ʻû����x�_�Y�X�\�t������������������������������������������������)�5�J�N�W�Y�T�N�B�5�)����������y�������ſҿܿ�ԿĿ������������u�v�s�y�N�Z�g�l�g�]�Z�N�I�F�N�N�N�N�N�N�N�N�N�N�(�A�Z�g�l�m�h�f�Z�N�A�5�*���������(�����%�6�:�2�&�)����������ùõ������M�Z�f�s�����������������������f�W�D�A�M���a�z���������������m�T�;�2������������/�T�a�h�j�a�H�B�/�"����������������	�/�#�/�<�=�>�<�3�/�#� ���#�#�#�#�#�#�#�#�׾�������۾׾־վѾ׾׾׾׾׾׾׾׾�"�7�A�B�?�9�/�)��	���׾;������׾�ŔŠŭŹ����������ſŹŭŨŠŜŘŗŐōŔìù��������ûùìèàÙààìììììì�5�N�[�w�w�t�i�c�`�[�O�B�)�������)�5�������/�B�K�B�/�"�	�������������������5�N�T�g�s�����������Z�N�A�-�����)�5�T�`�y�������������m�`�G�;�4�-�,�>�B�G�T�f�t�w�{�z�q�r�{�s�f�A�(���(�A�G�Q�b�f�zÇËÇÂ�z�q�n�f�i�n�q�z�z�z�z�z�z�z�z��������������ݿѿͿ̿οؿݿ�����Ŀѿݿ߿ݿܿտѿĿ��������������������hƁƓƚƧưƴƭƧƢƚƁ�u�\�O�I�K�O�`�h���	�"�/�4�;�H�H�:�/�"�!���������������s�����������s�m�p�s�s�s�s�s�s�s�s�s�s����*�6�A�F�M�C�6�*�������������������������#�0�?�F�D�;�#��
��������������ìù��������ûùìàÓÊÓÕàáìììì���������"�&�%���
������·¨¦¬¿������&�.�-�)�����������û���������āčĚĦĳ����ĿĳĚā�j�N�B�6�.�B�R�hā�\�uƚƳ���������������ƩƚƁ�m�[�X�\�f�s�����������s�f�d�e�f�f�f�f�f�f�f�f��������������߼���������ｅ�������������������������������������������������������������������������������y�����������������������y�t�l�d�e�b�l�y�����
�#�0�<�E�I�^�U�I�<�0��
����������EuE�E�E�E�E�E�E�E�E�E�E�E�E�E�EvEtEuElEu�a�n�zÅÃ�z�z�n�a�`�_�Z�a�a�a�a�a�a�a�a�!�-�7�:�D�E�:�7�-�)�!������!�!�!�!�ѿؿݿ߿ݿܿѿѿĿ������Ŀƿѿѿѿѿѿ��������������������������������������������	��� ��	�����������������������������������������������������#�/�<�H�U�`�c�V�H�<�/�#��
���
���r���������ƺɺѺպɺ��~�r�^�N�:�9�B�Y�r�нӽֽ׽սнʽĽýýĽĽннннннн��/�<�C�H�K�P�H�<�0�/�&�*�/�/�/�/�/�/�/�/��������������������������}������������4�@�M�T�Y�T�M�@�4�+�'��'�)�4�4�4�4�4�4D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�ED�D�D�D�D�D�D�D�D�D�D�D�D�DӼ����������߼����������� 4 ; 8 L 7 9 Y x S ! < ^ W A E - < N J A @ N Y � 1 B   A x . Y : = Y k . D F O 5 8 P M 4 G 5 Y 8 T @ , D ` L 1  ? K K P : F s D 6 # ; Z n    '  �  (  �  �  �  �  �  �  �  !  �  r  ~  �  h      ,  �  �  �  O  ;  W  c  �  �  \  �  /  E  w  �  d  �  �  B  �  <  �  �  �  �  �  �  r  f  �  �    -  k  �  �  �  �  j  I  �  �  �  �  w  '  �  �  �  P��9X�T���49X��o�D����o=�^5%   =t�<���<D��<o;�o<���>1'<ě�=��=\)<#�
<��=��T=49X=�{=\)<�9X<u=�t�<�<�9X=��=��`=e`B=ȴ9='�=o=�w<���=]/=8Q�<���=�{=�+=<j=�E�=���>   =�v�=�P=�w=0 �=,1=m�h=u=� �=P�`=e`B=}�=�+=e`B=�7L=ȴ9>A�7=���=ȴ9=�=��m>t�=�l�=�;dB%�B"G�Be�B�yB��B$
B�mB	��B1�B ��B%��B�B4�B��B0FB�(B��BJdB��BB��BSBY�A�|>B�(B4��B�1B 	BDB��B�B�B�B�KB�B�FB�B�RB
%�B�@B�GA���B!ØB%�BZ�B��B�9B�B$��B*1�B&��B,�uB�&B{�BB�qA��@B&B�B(�LB�:B�@B�B��B��B?&BV'Bp6B�?B%�vB"@�B@�B��B��B$"�B�\B	�ZB?�B ��B%��B�xBI�B?�B#�B�
B��B��BʟB?�B�kB7 B��A�u�B�UB4�`B��B 4&B��B��B�B��BJ�BL�B��B�yB��B�B	�AB��B�A�?$B"@wB@HB5�B@�BE}B=(B$�(B*=�B'rB,�[B��B��BʃB�rA��,BȻB��B)<B��B�B	�B�kB̶B@�B?�B��B�UA+�\@@.A���A�A��c@�Q�A�kKA���AdgdA4Wp@��RA���AĶ[A5aE@�@�A��JA��As��A�W�A�&�A���AF�gA�%A�4TA�RAT^�AX�JA��A�lA��XA�"hA��Ai�vA>N�A�:�A�_Ayi�B'$A�8AD�AA�OfA�߀A�`EA� �A�o�A�SB'sACԤA�A��@�]A?�A���C�)zAǦJ@r�pAz	�A���A���@Yx;Am@�[A)�A�|F@���@�2C��SC�/1A+SA,y[@�tA��.A�rrA���@� �A؅ AˤVAd��A5�@�VAϔ�A�R@A3�@��A�~
A�}At�oA�o�A���A҇AG$�A��jA�v�A��AT��AX�#A���A�f�A��hA�ffA��Aj�A=N�AȆ�A���Ay�KBB�A�~!AD=A�ktA�|�A˗�A�g~A�P?A� >B,lAD+.A�|A Β@�6LA	�A��pC�7dAǓ[@s�.Az�VA�u�A��]@[�A���@��A)N[A�K@�@�R[C���C�9�A��         
            c      *                  �      #            J   $   L            ;         A   T   &   O                        :   %      ;   ,   ]   =                     .   
   
                  v               0      
                     )      '   !               1         #      %   /      ?   )         )         '   ?   %   -   #               #               %   !   '   /                     !                        +                                                                                          )   )         !            9         #               #               #   !                                                                        Nk�N���N�]N���N��+N~��Oa[?N���O�S�O�$ZN���NODrNT��Nt�O�rbO��O	�YO��sN�;O�'O`+O1�P&�O�ЉN(�;N6�O�sNÛ+NDwOG��P�q`OyV�O�ҫO��N-��OJ��N�&�Ok�O���N�TOa�O'ǕN��O�ΡO��O��Oe��NEsNe�CNm�N @�Nd��O�{$O_�N���N��N�PQO!�3N5�N{>�O��LO��MN2BXNs(�N�&N�֐N�D�NxaxN-�  0  D  {  b  J    8  �  �  �  T  �  z  $  '  �  �  �  4  �  Z  0  3  V  �    �  �  �  �  �  �  �  g  r  �  �  �  %  {  �  t  (  �  �  �  ,  X  @    �  �  �  �  v  �  4  �  p  c  &  �    �    	9  p  �  �������ͼ�C����
��t��o=+��`B;�`B;o:�o%   ��o:�o=�%;�`B<u<t�;�`B<T��=8Q�<�j=#�
<49X<T��<T��<�/<��
<�o=<j<�/<��=H�9<��
<�1<ě�<�j<�/<ě�<ě�=<j=<j<�=#�
=C�=�O�=y�#=+=\)=�P=��=P�`='�=H�9=,1=@�=L��=L��=L��=]/=�t�=��`=��-=�9X=�^5=Ƨ�=Ƨ�=Ƨ�=���! #0:710.#!!!!!!!!!!��������������������!##/<@F@</*#!!!!!!!!'/0<HUVUNQH<:/''''''lnsz���������zxnllll����

�������??BN[gtz}}{wtgc[NEB?dbcdfgoqttx|ytogdddd
#/<CLSYYUQH/#
�������������������� #/01<AD<90#��������������������NCOS[hlmh[SONNNNNNNN!#$������������������������������������� 
 
����ltw}��������������tl72:<=HLPH<7777777777�����),8971)�('&-/<HOUZ^VUH=<:/((XXY[[bht�������tqh[X������� #�����"16;EHEJG;."���������������������������������������������):EJLI@6��lmqz��������zpmmllll��������56<BN[grtvuog][NBB:5yrt�����.55������y������������������������+6;<75)����{������������������{��������������������
#/1<BHHD</#
snmrt}���������tssss��������������������TPR_g~���������{tg[T����������������������������������������:99;@HT\abeb]WTIG?;:��������������������		 )5BN]^[TIB5)	����������������������������������������	)5:?>=93);89:<@HRSMH<;;;;;;;;�
#$)&#
����������������������������DCHIUVZYUIDDDDDDDDDD�������������������������#&'+*' ���������
""
���704<HUUZVUH<77777777srtu����������ztssssRT\ajmzzzmaUTRRRRRR�����������������������������������������{nhnn{����������������������������������������)12.*���25BN[_[TNB5522222222������ ������������KOO[\htw����uth[XOK|}������������||||||������

��������#)+'#!���������������������ݽ����ݽнϽнٽݽݽݽݽݽݽݽݽݽݺ��������������������������������������������������������������������������������T�_�a�m�h�b�a�[�T�H�?�F�H�N�T�T�T�T�T�T�����������������������������������������'�4�@�G�M�P�M�@�6�4�0�'� ��'�'�'�'�'�'�B�O�\�g�j�e�\�O�B�6�)�����"�-�6�;�Bàìù����������üùìàÓÐÓÚàààà�`�m�s�x�x�v�f�T�G�;�.�"�����"�7�W�`���(�4�A�D�F�@�5�(�������������������������������������z�|�����������������������������������������������������<�H�O�U�\�]�U�H�?�=�<�;�<�<�<�<�<�<�<�<��(�2�3�(��������������������ûлֻлƻ����������z�r�t�y��������������������������������������������������)�5�B�M�N�Q�N�I�B�5�)����	���������������ο׿ؿѿȿ������������y�{�z���N�Z�g�l�g�]�Z�N�I�F�N�N�N�N�N�N�N�N�N�N��(�5�N�Z�^�[�\�S�N�5����������������������
���������������������s�����������������������t�s�h�f�_�g�s�;�a�z�����������������m�a�T�H�8�/�"�'�;�/�T�a�h�j�a�H�B�/�"����������������	�/�/�<�<�=�<�2�/�%�#��#�#�/�/�/�/�/�/�/�/�׾�������۾׾־վѾ׾׾׾׾׾׾׾׾׾����%�5�6�3�-�"��	���׾Ѿžľʾ�Źż��������ŻŹŭŢŠŜŜŠŭŭŹŹŹŹìù��������ûùìèàÙààìììììì�)�5�B�F�M�O�K�B�=�5�)���������)�������	��/�>�G�D�/�"�	�����������������N�Z�g�q�|����s�Z�N�A�9�5�,�%�(�,�5�A�N�m�y�����������y�m�`�T�G�F�@�?�?�G�T�`�m�f�t�w�{�z�q�r�{�s�f�A�(���(�A�G�Q�b�f�zÇËÇÂ�z�q�n�f�i�n�q�z�z�z�z�z�z�z�z���������������ݿοͿϿڿݿ�����Ŀѿݿ߿ݿܿտѿĿ��������������������h�uƁƍƚƧƬƱƧƚƁ�u�\�U�O�O�T�\�e�h���	�"�/�4�;�H�H�:�/�"�!���������������s�����������s�m�p�s�s�s�s�s�s�s�s�s�s��������*�7�8�7�*��������������������
��#�0�7�6�0�'�#��
����������������ìù��������ûùìàÓÊÓÕàáìììì�������
���"�#�!��
��������²°µ¿������#�,�)�����������ý������������tāčĚġĭĴĴĳĦĚčā�t�j�[�Y�`�h�tƎƚƧƳ������������������ƳƧƚƉƁƅƎ�f�s�����������s�f�d�e�f�f�f�f�f�f�f�f��������������߼���������ｅ�������������������������������������������������������������������������������y���������������y�u�u�t�y�y�y�y�y�y�y�y�����
�#�0�<�E�I�^�U�I�<�0��
����������E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E{EzEE��a�n�zÅÃ�z�z�n�a�`�_�Z�a�a�a�a�a�a�a�a�!�-�7�:�D�E�:�7�-�)�!������!�!�!�!�ѿؿݿ߿ݿܿѿѿĿ������Ŀƿѿѿѿѿѿ��������������������������������������������	��� ��	�����������������������������������������������������#�/�<�H�U�`�c�V�H�<�/�#��
���
���e�r�~�������������������~�r�]�R�L�L�V�e�нӽֽ׽սнʽĽýýĽĽннннннн��/�<�C�H�K�P�H�<�0�/�&�*�/�/�/�/�/�/�/�/���������������������������������������4�@�M�O�V�R�M�@�4�.�'�"�'�,�4�4�4�4�4�4D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�ED�D�D�D�D�D�D�D�D�D�D�D�D�DӼ����������߼����������� 4 ; % L 7 9 / x W  = B W A . + . A J J   1 ; � / B  ( x   ` # 4 Y k - D C O 5 / C M 4 J * B 8 T @ , R ` G 1  ? K K P : , s D 6   ; Z n    '  �  �  �  �  �  �  �  �    �  p  r  ~  e  /  2  �  ,  �  /  n  �  ;  F  c    �  \  �  �  �  �  �  d  �  �  �  �  <  �  z  �  �  �    �  f  �  �    �  k  �  �  �  �  j  I  �  �  �  �  w    �  �  �  P  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  0  )  !        �  �  �  �  �  �  �  h  P  :  $     �   �  D  A  >  :  6  1  *         �  �  �  :  �  }  #  �  h    p  s  v  w  x  y  z  w  r  k  d  Y  L  ;  $    �  �     h  b  Z  Q  I  A  :  7  4  0  -  -  /  2  4  6  8  9  :  ;  =  J  I  H  G  @  7  .  #      �  �  �  �  �  �  �  K   �   �    
     �  �  �  �  �  �  �  �  �  q  W  ;      �      �  
  
�  I  �  �  %  5  7  (    �  �  8  
�  	�  �  �  L    �  �  �  ~  j  W  D  2         �  �  �  �  �  �  �  n  8  A  {  �  �  �  �  �  �  �  �  ]  #  �  �  _     �  F  �    p  �  �  �  �  �  �  �  }  c  L  ,  	  �  �  �  2  �  w    O  O  O  Q  S  T  O  J  E  =  2  $       �  �  �  �  �  (  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  z  r  j  b  Z  R  K  C  <  5  .  '  #  !        !  #  &  $  1  =  "  )  9  C  H  T  d  �  �  �  �  �  P    �  �  N  	Y  	�  
r  
�  @  �  �  �    '  
  �      
�  	�  �  u  �  _  �  �  �  �  �  �  �  �  z  b  G  +  
  �  �  ^  �  �    �  )  P  j  �  �  �  �  �  �  �  f  <    �  `  �    W  �   �  t  �  �  �  �  �  �  �  �  v  ^  ?    �  �  c    �  L  �  4  ,  $          �  �  �  �  �  �  �  �  �  �  t  d  U  \  t  |    �  �  {  r  o  w  w  h  M  )  �  �  \  �  :   �  �  �  Y  �  D  �  �  *  J  W  W  P  C    �  L  �  �  �    �      *  ,  (  %  /  ,      �  �  g    �  7  �  O    l  �  �  �      (  /  /    �  �  g    �  �  ?  ^  �  �  V  O  I  >  #  �  �  ~  9  }  l  J    �  d  �  h  �  5   �  �  �  �  �  �  �  �  �  ^  %  �  �  r  4  �  �  o  ,   �   �                        �   �   �   �   �   �   �   �   �   �  �  �  �  �  �  �  �  �  �  �  y  R  '  �  �  0  �  <  �  �  ~  z  z  {  �  �  �  �  �  �  �  �  l  B  
  �  �  [  .    �  �  �  �  �  �  �  |  j  X  C  )    �  �  �  �  �  �  j  �  .  g  �  �  6  n  �  �  �    V     �  �    k  �  u  :  L  �  �  ~  B     �  �  u  �  [  @    �  �  R  �    &  �  �    B  e  |  �  �  �  x  e  U  D  *    �  �  W  �    b  �  �  �    H  w  �  �  �  �  R  
  �  H  �  b  �  �      g  ]  W  G  1    �  �  �  �  �  �  {  J    �  �  �  d  J  r  z  �  �  �  �  �  �  �  �  �  �  �  �  �    w  n  f  ^  �  �  �  �  �  �  x  f  R  >  (    �  �  �  �  I  �  f   �  �  �  �  �  |  o  c  Y  R  J  B  ;  3  ,  %        
    �  �  �  �  �  �  �  �  a  =    �  �  �  ^  '  �  �  7  �  %    �  �  �  �  �  �  �  �  x  E  (    �  �  K    �  �  {  y  w  t  r  p  n  l  j  h  f  e  d  b  a  `  _  ^  ]  [  �  A    �  �  �  �  �  �  �  �  b  &  �  _  �  :  �  �  �  �  �  (  G  ^  j  l  q  s  o  Y  2  �  �  2  �  M  �  ^  q  (    �  �  �  �  f  <    �  �  u  '  S  G  6  "     �  %  �  �  �  �  �  �  �  �  �  a    �  |    w  �     "  &  @  �  �  �  �  n  M  #  �  �  u  &  �  e  �  |  #    �  �  �  	�  
�  4  �  �  6  h  �    H  �  �  4  
�  	�  	    �  =  m  P  w  �  �  �  �    &  +  ,  '    �  �  |  �  W  �  �  �  X  U  Q  M  J  G  F  E  D  C  B  ?  <  9  7  A  O  ^  m  {  @  4  '         �  �  �  �  �  �  �  �  z  h  U  B  /                  �  �  �  �  �  �  �  �  m  H     �   �  �  �  �    v  n  h  c  ^  Y  T  P  K  G  B  ?  =  ;  8  6  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  6  �  �  3  �    {  t  e  P  C  #  �  �  �  j  3  �  �  �  �  ]  '  �    r  �  �  �  �  �  }  R     �  �  c  =    x  �  �  �  �  v  l  b  T  F  ;  0  %        �  �        !  )  0  8  �  �  �  �  ~  v  n  e  Z  M  @  1    
  �  �  �  x  6  �  4         �  �  �  �  z  c  L  3        �  �  �  v  I  �  �  �  �  �  �  �  u  V  1    �  �  �  o  L  #  �  �  &  p  k  f  a  ^  \  Z  T  K  B  6  (      �  �  �  �  |  O  c  L  4    �  �  �  �  w  Q  (  �  �  �  k  %  �  �  H    &      �  �  �  �  �  r  D    �  ~  7      �  �  c    !  �    \  �  �  �  o  G    �  h  
�  
]  	�  �  -  "  T  �                
  	    	            $  (  -  1  �  {  f  P  9  #    �  �  �  �  �  �  �  �  m  V  >    �        �  �  �  �  t  G    �  �  q  1  �  �  j  $  �    	,  	4  	8  	9  	4  	*  	  	
  �  �  �     �  =  �    �  �  6  .  p  X  6    �  �  ]    �  [  
�  
�  
  	F  f  a  S  ;    �  �  �  �  _  .  �  �  �  ^  -    �  �  "  �  �  �  9  �  '  �  �  �  �  �  �  �  o  Y  C  -       �  �  �  �  �  �  �