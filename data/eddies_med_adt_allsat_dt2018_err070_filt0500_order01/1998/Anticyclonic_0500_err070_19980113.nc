CDF       
      obs    =   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type                     	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?h�t�j~�   max       ?�5?|�i      �  �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       M�(>   max       P���      �  �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       �<j   max       >n�      �  �   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>�=p��
   max       @E���Q�     	�   �   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?�
=p��    max       @vd�����     	�  *   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @%         max       @M@           |  3�   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @�9        max       @�`          �  4   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       �t�   max       >I�^      �  5   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A��C   max       B+�M      �  5�   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A�~�   max       B+Δ      �  6�   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       ? �   max       C� J      �  7�   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       ?��   max       C��      �  8�   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          w      �  9�   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          C      �  :�   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          3      �  ;�   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       M�(>   max       P]e�      �  <�   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�_��Ft   max       ?��S���      �  =�   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       �8Q�   max       >n�      �  >�   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>�z�G�   max       @E��\)     	�  ?�   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?�
=p��    max       @vd�����     	�  I   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @!         max       @N�           |  R�   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @�9        max       @�`          �  S   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         D�   max         D�      �  T   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�e+��a   max       ?�ᰉ�'S       T�   
   	                                 
      Q   %            F   _   Y   E            1   $   	   (            (   +   v      j               1      K         &               :            "      I      O��N���N^��NZ��N��qND��N�kxN�v�N�h�N��OgdN���N�M�>cP!��P�NT�N�l�O.~ZPa�pPde�P��KP���NsSN�*Nq�O��|O�зNB��O�G�N��OL�Ow�&P �LOwP�PS�*N �PZ�RO�NR�N�یO!o/O�l�O��P:�OY8N�l�OP�nO?_�N�P#M�(>NyOO�eTN.n�N�һN���N��NJ�+O�hN7P�NL�<j�\)������j��j�u�o��`B���
�D��%   %   :�o<o<t�<t�<t�<t�<D��<e`B<e`B<�C�<�C�<���<���<�1<�j<�`B<�`B=o=+=t�=t�=�P=�P=��='�=,1=0 �=8Q�=@�=Y�=m�h=u=�o=��=�O�=���=���=���=���=��-=�Q�=ě�=��=�"�=�`B=�l�=��=��>n�����������������������������������������MNUUVW[`egjjg][NMMMM~z~���������~~~~~~~~ #&/<>EF</# !#09<<<40#        ()5>BNSWTNB51)((((((%()-6BNOOUZUOB6)%%%%��������������������:9<BHRUV\_UHD<::::::������������nmst��������~tnnnnnnnpt~�����tnnnnnnnnnn��������������������"(.7Bmt{xj[OB:"NJQXh�����������thXNcdhtwuxthhccccccccccttz�����������}{ztttE==BHQUansuronhaUQHE��������������������
/CHPTXYWOH< �������).5)�������������*
������������������������������� 

������������������������������~{zz~��������������~
"/;DIH?8/+"	,,+.0<FG?<20,,,,,,,,USTKH<0#
#/<IU$)-2)&	����������������������������������������TSVan�����������naVT������������������������;LNkmc\VD)���������������������������5HNPLEA;5)���������
#.30#
�����������������������8459<HTU_aUMH<888888DEFHLSTUZacmsxmaTJHD���'-22/)�����������������������������������
�������������������������������

 ���� !"(/3;HTYXXUPH;/," U[abflmuz}������zmaU��������������������^bkn{||{nb^^^^^^^^^^����������������������������������������������������������/+&%/<@EH?</////////��������������������������������������zyqswz������zzzzzzzzz}����������������{z����������������HHD<305;<=CHHHHHHHHH������	������������������������������=�E�H�D�=�0�$���$�0�8�=�=�=�=�=�=�=�=�U�V�b�n�{ŇŌŇ�{�v�n�b�Y�U�U�U�U�U�U�U�a�n�z�~À�z�w�n�d�a�`�^�a�a�a�a�a�a�a�aD�D�EEE
EED�D�D�D�D�D�D�D�D�D�D�D�D�x�����������������x�u�w�x�x�x�x�x�x�x�x����������������ŹųŴŷŹ�������������ƺ����"�#�'�'�'����������������ܹ�����������ܹϹù��ùùϹԹܹ�E\EiEqEuE{EuEqEiE\EPECEMEPEVE\E\E\E\E\E\��*�-�6�4�3�*���������������L�Y�e�i�o�j�e�Y�L�@�@�=�@�E�L�L�L�L�L�L�������ûŻû������������������������������!�%�&�!�����������������'�M�Y�f�o��������r�Y�4����ݻ��A�Z���������Z�$�����ݽ��������A�l�x�}��x�l�_�Y�_�_�l�l�l�l�l�l�l�l�l�l�F�F�S�W�S�H�F�;�:�8�-�*�!��!�-�:�F�F�F���������������þ�������������{�|����������������������p�Z�A�(���5�A�L�Z�g����/�;�T�e�j�k�a�X�;�"����������������������+�U�j�n�f�U�0�ĽĽķĘėĜ�����񾌾��ʾ�	�.�A�G�.���㾼���a�V�`�q�s�~�����'�-�-�-�'�#����	���������4�@�B�M�N�M�E�@�;�4�2�.�4�4�4�4�4�4�4�4�����ʼμּʼʼ�������������������������ù������������������������ùóìæßÞù�;�T�a�u�{�x�m�a�T�;�/����������	��;�f�r�����������r�n�f�b�f�f�f�f�f�f�f�f�l�`�G�;�9�8�;�?�N�T�_�m�y�����������y�l���������������������{�y�m�l�m�q�y������"�/�;�H�O�W�a�m�y�m�a�X�T�H�;�0�"���"������'�,�0�-�'��������������5�N�a�j�r���������g�N�?�(�$�����������������������������������������������������$�3�O�h�s�O�6�������ýùú�����
�������
��������
�
�
�
�
�
�
�
Ƴ����$�-����������ƚƁ�h�]�a�pƂơƳ�������ʾ׾����ݾʾ���������u�u�x�����;�H�O�S�K�H�;�2�0�;�;�;�;�;�;�;�;�;�;�;���������	���������������������������tčĚĠĦĳķĳĦĚč��t�h�\�_�e�h�j�t���Ŀÿſſ������m�`�T�L�H�G�L�_�y������ÇÓàìùý��������ùìàÓÇÃÀÁÄÇ����/�<�H�U�\�a�c�a�T�H�/�#�
�����������������������������������y�q�q�m�q�y�����������������������������������
���#�,�0�7�;�<�0�#��
�������������
�<�I�b�n�{ŇōŔŗŇ�{�n�b�U�I�D�>�;�:�<���������������������������������������Ѻ�����	��������������������������������������������������������������������Ƽʼּ���.�:�=�:�)� ��ּ����������������������������������~�v�����������������b�o�{ǈǔǏǈ�{�o�b�_�a�b�b�b�b�b�b�b�bEuE�E�E�E�E�E�E�E�E�E�EuEnEjEuEuEuEuEuEuD�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D��4�@�M�Y�f�j�f�b�Y�M�B�@�4�4�4�4�4�4�4�4�l�x�}�}�v�o�_�N�F�=�:�-�'����!�F�_�l�-�2�6�2�-�!�����!�"�-�-�-�-�-�-�-�-���������*�*�*���������� 3 7 o , = 3 ^ K T C ?  > u R b M M  g \ C O : X 1 * d P , ' Q J R C R I O M H . ^ E ,  B G 4 Q @ r ? Y b = : m _ B Q V    \  �  �  k  �  Y  �      �  5  �  :  X  ?  l  !  �  i  �  =  u    �  I  u  F    p    �  �    �    �  Q  $  k  p  �  �  W  `  {  �  6  �  �  �  5  �  ?  h  �  �  �  �    q  D�t���/��9X����t��#�
��o%   ;�`B<e`B<o<�1<#�
<49X=�Q�=<j<D��<D��=��=���=�;d=��=� �<ě�<�9X<�/=�hs=�o=�P=�hs=0 �=T��=Y�=���=���>�w=D��>�+=��=P�`=�C�=�o=�"�=�9X>O�=�9X=�{=�l�=Ƨ�=�9X=��=�j>�+=���=�l�>o>z�=��>I�^>	7L>�wBU\BTB�*B
�bB?�B%��B�YB�BGBg?B�~B��B�EB *�B�cBQNB�B�`B�B�rBmB��Bn�B"IB#��B"�HB)A��CB&WBLUB�YB�B!�B��B��B��B,>BR3B$))B ]/BVA�.�B�hB!��B�`B+�MB|�A���A��BTB(3=B̅B!B
��B�+Bq�B��BNB.B%IB�tByBF�B��B
��B~�B%��B��B6�B%�B>�B��B��B��B >�B�dB�hB�B�B<TB�pB�.B> B@VB"=�B#ǫB"��B@�A�~�B&?�B�B��B�B �ZBB�B��B@NB�mBBB#�$B ;�B=�A��B��B"?0B��B+ΔB?�A�~�A���B�B(>B��B?�B
� BS�B��B��B=$B@BD"B�A�l�B
V�A�ƸAǋC�?�@��wA�bL?n��? �C��A�d?�<@�W�@fÌ@���A<��@�`�@{A�AJ))A�T�A�w%A�ЎAOQ @ƽ�@�]-@�Aσ�A��@�:+Ah�.Ao�MA�G�?UrA�?�A��A�OA���B��AJ�#A�FA��A�}An��A˧�A���A�CA�&A� A泥A��J@T�A���A�A��pB"/C� JC���@�:�@���@p�^A��GA��B
:�A�dAǂ�C�G_@��A�7m?}�l?��C��5A��D?�İ@��d@f;@�_�A>�@��W@{c�AJ��A�Z�A���A�eAM.@�4,@��@�L*A�~hA�c�@� �Ah�=Ap��A��C?C�A��A�A҂A�yB�OAK 7A�:IA�l:AݑPAm�A�d�A#A�A���A�^A�J�A���@TyA���A��A�ZBA	C��C���@��)@���@lx�A�}y      	                           	      
      R   &            F   `   Z   F            1   %   
   (            (   +   w      j               2      K         '               :            "      J                                                   +   1            5   3   7   C               %                  )      1      1               %      %                        #                  %                                                   %                  /   3   -                                 )            %                     !                        #                        O>�N���N�NZ��N��qND��N�kxN�v�N�h�N��OgdN-I:N�M�>cO�,�O���NT�N0�oO{O�v�P8��P]e�P#HtNsSN�*Nq�OM]O�LyNB��O�G�N��OL�Ow�&P �LOV�O�"N �O��O	-zNR�NM�HO!o/O���O��O��O7�N�l�O&�IO/��N�P#M�(>NyOO�eTN.n�N�һN���NE�NJ�+O�hN7P�NL  �      ^  G    U  �  �  P  �  s  <  �  �  4  �  �  �    �  	w  �  6  �  �  �  �  @      �  4  @     >    
�  �    �  8  �  �  :  x  �  �    (    �  	b  9  8    �  b  \  G  ��8Q�\)���ͼ�j��j�u�o��`B���
�D��%   ;�`B:�o<o<�`B<���<t�<#�
<�C�=��<���<�`B='�<���<���<�1=\)=C�<�`B=o=+=t�=t�=�P='�=�1='�=�\)=Y�=8Q�=Y�=Y�=}�=}�=���=�C�=�O�=��
=���=���=���=��-=�Q�=ě�=��=�"�=�F=�l�>J=��>n�����������������������������������������WVWY[]bghhg[WWWWWWWW~z~���������~~~~~~~~ #&/<>EF</# !#09<<<40#        ()5>BNSWTNB51)((((((%()-6BNOOUZUOB6)%%%%��������������������:9<BHRUV\_UHD<::::::������������tqt{�������tttttttttnpt~�����tnnnnnnnnnn��������������������')18B[htqprnhb[OB;*'XRQRT[ht��������tf_Xcdhtwuxthhcccccccccc�}|y����������������BBHJU\adnqronkbaUJHB����������������������/<BKSTQQNH<�������%+..-��������������������������������������������� 

��������������������������������������������������"/;?EDA=5/&",,+.0<FG?<20,,,,,,,,USTKH<0#
#/<IU$)-2)&	����������������������������������������TSVan�����������naVT������������������������)5BIKJE5)��������������������������)5ADB95)������
 
	�������������������������;78<AHJUUUUH><;;;;;;DEFHLSTUZacmsxmaTJHD����%),00-)����������������������������������������������������������������

 ����$+/7;HRTTUSMH@;4/%"$]\_bcgmxz|������zma]��������������������^bkn{||{nb^^^^^^^^^^����������������������������������������������������������/+&%/<@EH?</////////�������������������������� �������������zyqswz������zzzzzzzz�}{��������������������������������HHD<305;<=CHHHHHHHHH��� ���������������������������������=�E�H�D�=�0�$���$�0�8�=�=�=�=�=�=�=�=�b�n�{ŇŊŇ�{�q�n�b�^�Y�b�b�b�b�b�b�b�b�a�n�z�~À�z�w�n�d�a�`�^�a�a�a�a�a�a�a�aD�D�EEE
EED�D�D�D�D�D�D�D�D�D�D�D�D�x�����������������x�u�w�x�x�x�x�x�x�x�x����������������ŹųŴŷŹ�������������ƺ����"�#�'�'�'����������������ܹ�����������ܹϹù��ùùϹԹܹ�E\EiEqEuE{EuEqEiE\EPECEMEPEVE\E\E\E\E\E\��*�-�6�4�3�*���������������L�Y�_�e�i�e�_�Y�M�L�D�L�L�L�L�L�L�L�L�L�������ûŻû������������������������������!�%�&�!���������������'�4�M�Y�f�u�z�s�f�Y�4�'���������'�4�M�Z�f�s��������f�Z�4�(������(�4�l�x�}��x�l�_�Y�_�_�l�l�l�l�l�l�l�l�l�l�!�-�:�F�S�S�S�F�D�:�.�-�!� �!�!�!�!�!�!�������������������������������������������������������������g�Z�P�W�_�g�s�����"�;�T�a�e�e�]�T�;��	�����������������"������#�W�`�d�]�I�0���������ģğĤ���񾥾��׾������	�𾾾�����h�g�q���������'�-�-�-�'�#����	���������4�@�B�M�N�M�E�@�;�4�2�.�4�4�4�4�4�4�4�4�����ʼμּʼʼ�������������������������ù����������������������������øíìóù��"�;�T�a�m�r�v�r�m�a�T�H�;�/�!���
��f�r�����������r�n�f�b�f�f�f�f�f�f�f�f�l�`�G�;�9�8�;�?�N�T�_�m�y�����������y�l���������������������{�y�m�l�m�q�y������"�/�;�H�O�W�a�m�y�m�a�X�T�H�;�0�"���"������'�,�0�-�'��������������5�N�a�j�r���������g�N�?�(�$������������������������������������������������������!�+�0�0�*������������������
�������
��������
�
�
�
�
�
�
�
ƧƳ�������	��	�������ƳƎƄ�{�~ƈƎƧ���������ɾƾ�������������������������;�H�O�S�K�H�;�2�0�;�;�;�;�;�;�;�;�;�;�;�������������������������������������tčĚĠĦĳķĳĦĚč��t�h�\�_�e�h�j�t���������¿¿��������m�`�T�O�J�J�O�`�y��ÓÙàìù��������ùìàÓÇÄÁÃÇÉÓ�#�<�H�U�^�_�]�U�H�<�/�#�
������������#�y�������������������������~�y�s�r�q�v�y��������������������������������#�(�0�4�6�0�#� ��
���������������
��<�I�U�b�n�{ŊŔŕŇ�{�n�b�U�I�E�?�<�;�<���������������������������������������Ѻ�����	��������������������������������������������������������������������Ƽʼּ���.�:�=�:�)� ��ּ����������������������������������~�v�����������������b�o�{ǈǔǏǈ�{�o�b�_�a�b�b�b�b�b�b�b�bEuE�E�E�E�E�E�E�E�E�E�EuEnEjEuEuEuEuEuEuD�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D��4�@�M�Y�f�j�f�b�Y�M�B�@�4�4�4�4�4�4�4�4�F�_�l�x�|�|�u�n�_�L�F�:�-�)�!����!�F�-�2�6�2�-�!�����!�"�-�-�-�-�-�-�-�-���������*�*�*���������� 4 7 z , = 3 ^ K T C ?  > u I C M f   5 Y B M : X 1 ! Q P , ' Q J R 9 0 I C 1 H 0 ^ @ &  A G ) P @ r ? Y b = : ` _ @ Q V    ?  �  N  k  �  Y  �      �  5  B  :  X  �  �  !  {    �  �  �  *  �  I  u  �  G  p    �  �    �  �  �  Q  o  /  p  o  �  �  A    �  6  c  �  �  5  �  ?  h  �  �  �  �  �  q  D  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  �  �  �  �  �  �  �  �  �  ~  k  V  ?  %    �  �  �  x  Q        �  �  �  �  �  �  �  u  b  N  2    �  �  �  �  �              �  �  �  �  �  �  �  f  L  =  1  %      ^  V  N  F  >  5  )        �  �  �  �  �  �  �  r  ^  J  G  5  #      �  �  �  �  �  �  ~  b  D  $     �  �  b          
    �  �  �  �  �  �  �  �  �    q  ]  G  0    U  T  S  S  R  Q  Q  P  O  N  L  I  F  C  ?  4  %      �  �  �  �  �  �  �  �  �  �  �  �  �  m  >    �  �  �  �  �  �  �  �  �  �  �  q  `  P  >  *      �  �  �  �  �  �  ~  P  A  H  8        �  �  �  �  �  ^  0  �  �  �  U    �  �  �  �  �  |  t  j  _  W  N  H  C  @  I  S  c  w  ~  r  e  �  �  �  %  B  \  n  t  n  W  '  �  �  d    �  A  �  F  �  <  :  8  5  2  /  +  $        '  s  �  �  �  �  �  z  q  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �    a  D  &  U  �  G  �  �  �  �  Y  0  �  �  z  2  �  �  j  �  (  �    �  �  �  
  0  4  0  %      �  �  �  �  t  $  �  j  �  S  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  a  4    �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  {  l  ]  N  ?  0  !  �  �  �  �  �  �  �  �  �  �  e  *  �  �  d     �  �  ]  b  K  �    S  �  �  �    �  �  �  �  X  	  �  �    }  �  �  �  �  �  �  �  �  �  �  �    i  O    �  X  �    0    (  	)  	e  	v  	p  	Y  	9  	  �  �  G  �  �  :  �  b  �  o  �  �  �  D  [  _  b  m  �  �  �  w  U  7    �  ]  �  A  �  �  ,   |  6  6  6  6  6  7  7  8  8  8  8  8  K  l  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  s  d  T  E  5  &    �  �  �  �  �  �  �  �  �  m  X  E  2      �  �  �  �  {  2  T  q  �  �  �  �  w  ^  6  	  �  �  %  �  �  �  �  �  �  K  N  M  �  �  �  �  d  C    �  �  #  �  0  �    _  �    @  ;  5  2  0  5  <  :  3    �  �  �  }  Q  #  �  �  �  V       �  �  �  �  �  z  a  M  9    �  �  V    �  G  �  �        �  �  �  �  �  �  �  �  �  �  �  �  n  F    �  �  �  �  �  q  V  <  "    �  �  �  u  >    �  o    �  o  �  4    �  �  �  �  z  `  L  6    �  �  �  �  y  C    �  #  @  !    �  �  �  s  T  4    �  �  |  @  �  �  [  �  �  G  �  �  �  �  �  �  [    �  {  ,  �  �  \    �  �  �  �   �    �  �  �  �    +  <  8  &  �  �  /  �  
�  	�  �  r  �  <        �  �  �  �  �  �  �  �  �  k  Q  8       �  �  �  	�  
   
x  
�  
�  
�  
�  
�  
�  
�  
Z  
  	�  	V  �  5  q  M  �    L  e  �  �  �  �  �  �  �  �  �  r  T  1  
  �  �  y  ,  �    y  s  n  c  V  J  ?  5  +        �  �  �  �  �  �  �  u  �  �  �  �  �  �  �  �  �  �  ~  f  B    �  �  7  �  _  8    �  �  �  �  v  X  C  +    �  �  �  �  u  E    �  �  l  �  �  �  v  M    �  �  _    �  G  �  _  �  G  �  �  �    �  �  }  f  <  
  �  �  D     �  �  O    �  �  �  �  �  
�  %  6  9  '    
�  
�  
P  
  	�  	Q  �  Z  �        �  ]  u  w  w  x  v  v  t  n  c  V  G  3    �  �  S  �  t  �  7  �  �  �  �  �  m  R  2    �  �  �  X     �  �  v  7  �  9  �  �  �  �  �  �  �  �  i  5  �  �  a  �  `  �    T  �  �        �  �  �  �  �  �  z  U  *  �  �  k    �  #  �  �  (  '  &  !      �  �  �  �  �  �  v  b  O  <  (      �          �  �  �  �  �  �  �  �  c  C  #    �  �  �  �  �  �  �  |  b  B     �  �  �  w  `  E    �  �  K  �  �  ]  	b  	Q  	%  �  �  X    �  �  �  X  
  �  _    �  (  i  �  z  9  .  #        �  �  �  �  �  �  �  �  t    �    �    8    �  �  �  �  {  Y  6    �  �  �  �  g  F  (    %  >      �  �  �  �  _  7    �  �  s  7  �  �  x  0  �  �  �  �  �  �  �  �  �  u    
�  
$  	�  �  �       1  A  J  M  J  b  '  �  �  �  �  �  �  �  }  i  Q  7    �  �  W    �  {  G  [  O  6    �  d    �  U  
�  
T  	�  �  /  ^  j  +  �  �  G  "  �  �  �  �  }  e  I  %  �  �  �  F    �  |  6  �  �  �  �  X  5    �  �  �  v  L  !  �  �  �  Y    �  �  Z  