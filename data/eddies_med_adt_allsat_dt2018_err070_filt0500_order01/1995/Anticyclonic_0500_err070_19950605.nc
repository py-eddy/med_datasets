CDF       
      obs    7   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type                     	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?h�t�j~�   max       ?�j~��"�      �  �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       M�t�   max       P�F�      �  �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       �L��   max       =��      �  d   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?�z�H   max       @E��Q�     �   @   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?񙙙���   max       @vo
=p��     �  (�   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @(         max       @P`           p  1p   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @�9        max       @�-�          �  1�   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       �<j   max       >��      �  2�   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A�+X   max       B+�)      �  3�   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A�|T   max       B+�_      �  4t   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       ?��"   max       C�k�      �  5P   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       ?�_�   max       C�X      �  6,   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          o      �  7   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          =      �  7�   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          3      �  8�   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       M�t�   max       PVfC      �  9�   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?���%��2   max       ?�(����      �  :x   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       �L��   max       =��      �  ;T   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?(�\)   max       @E���R     �  <0   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?񙙙���   max       @vo
=p��     �  D�   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @!         max       @P`           p  M`   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @�9        max       @�@          �  M�   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         @�   max         @�      �  N�   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�IQ���   max       ?�&�x���     0  O�                  
   )   
      
         $      M      '   T            +   ;      (   ,      !      ?               V      o   K      5   O   9                     K   7   0            M���OJ�NF��N>^�O��N۴4Pt9N�NyuN꡹N�'�OTO���O�0Pr�QO��<O�eP�F�N���O2��M�wWO��bPh_�O��'O��PsxN�H^O���N�EP+N��
N�G�PyN��O�Y�N���O�.�Oy�O5�P��PGJOѸlO"ԜN�n�M�t�N"J)OLCN%��O�
�O��O��YO/I�N^Z&M�q	N�;�L�ͼ�j��t��D����`B���
��o�D���D����o;D��;D��;ě�<t�<#�
<49X<�o<�o<�t�<���<���<��
<�1<�j<ě�<�/<�/<�<�=o=o=+=+=C�=��=��=#�
=#�
='�='�='�='�=0 �=8Q�=<j=<j=H�9=H�9=Y�=]/=q��=}�=�v�=�x�=����������������������fgjqt���������tmhgff����������������������������������������fgqstx�����������tgf�������

������ztz���������������zz(%%&()6<BCBB>61)((((jfin{~���{qnjjjjjjjjxyz������������zxxxx����������������������� 

�����������������������������������������������������)N]^TI5$����2BNY[aa\VIB5)(y|{���������������~y��������*/-������%')6=BFMB762)%%%%%%���������������������������������������� #0IS\``]UNI0"����)27FHC5��������
#/29;<1/(#
�#<HUaw{slij_U<#����������������������������������������������������������������������������������.5KNPJTUNB)JIHKN[cggga^[NJJJJJJgjmtyz�����zvmgggggg #/<AKO\d\UH</##%(++# ������%),)#���{��������������?@AEN[gt�������[NIB?����
!$#
��������tomnrt������������tt�������#/586)
����������������������������������������ZY`dghot������{th`[Zyuurtrppz�������|{zy$#&)5775+)$$$$$$$$$$HIJO[f`[ZOHHHHHHHHHH������ ��������mnz����������zwnmmmm?=BO[htz�����{h[OGC? "*/:;HT\`abb[TH;" �)5BGJLKKIB5)��������������������� #%/;<?<6/#        "),*)&���������	����纤��������������������������������������ÇËÓàãäâàÓÇ�z�s�n�j�h�n�z�~ÇÇ��������������������������������������������������������������������������������������������������������������������������(�������������������������������������������s�Z�A�(���(�A�s���l�x���������������������x�m�l�a�l�l�l�l�x�������������x�l�a�l�o�x�x�x�x�x�x�x�x�T�a�a�m�s�z���|�z�m�a�T�H�J�L�S�T�T�T�T�A�M�V�Z�[�Z�M�A�A�4�(�$����(�4�6�A�A��(�4�A�L�B�4�(��������������������ĿѿٿڿؿѿĿ�������������������àìù������������ùðìàÓÇÆÂÇÓà�	�"�;�a�m�|�������z�m�T�H�?�/�%������	�������*�9�A�D�5�(����ؿƿοѿۿ��N�Z�g�s�q�s�������w�g�Z�N�(�����(�N��������/�A�Q�O�>����������������������zÇÓÔ×ÓËÇ�z�v�n�l�n�u�z�z�z�z�z�z��"�+�.�:�:�A�;�.�(�"��	��������� �	���������������������������������������������������ȼǼ�����������p�f�b�X�Y�r��0�I�U�bńŋ�{�b�0�����Ľĺ����������0�y���������������y�m�^�G�A�;�1�9�G�Q�`�y������������������ùëð�����������)�0�B�O�V�B�:�-���������������āčĖĘĐčăā�t�s�h�[�[�[�h�k�t�{āā�O�\�h�uƁƖƢƤƞƚƎƁ�h�\�O�E�C�F�L�O�*�6�<�9�6�1�*�������!�*�*�*�*�*�*ƎƚƳ�����������!�����ƚƆ�|�v�v�|Ǝ�0�=�I�V�Z�_�V�K�I�I�=�0�'�%�0�0�0�0�0�0ŔŠţŭŹźźŹŭŠŗŔŔŒŔŔŔŔŔŔ�M�f��������Ǿ�����s�Z�A�(�!���!�-�M�	�����	���������	�	�	�	�	�	�	�	���!�-�4�G�I�D�:�-���������޺����<�H�T�U�X�U�P�H�@�<�/�'�$�(�/�9�<�<�<�<�B�O�[�h�{ĄĀ�x�k�N�8�6�(�&�(�"�%�)�6�BD�D�D�D�D�D�D�D�D�D�D�DzDoDsD{D�D�D�D�D������	��"�%�.�+�"�"�"���	��������������	��"�2�3�8�=�>�6�/�"����������������M�Y�r�������r�U�J�4����������#�4�M�ʼּ���� ��������ʼ��������������ʻ-�:�F�S�_�b�l�u�o�l�_�S�F�:�-� ��!�&�-�:�F�S�_�l�x�����������x�l�_�S�K�F�:�9�:�.�;�G�K�K�G�;�9�.�+�.�.�.�.�.�.�.�.�.�.������
������������������H�O�U�_�a�c�a�U�H�<�/�#����#�$�/�<�HE�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�Eټ@�f�t�������z�r�@�4���޻������'�@���
�0�<�B�I�K�H�5�#��
������������������������������²¦�t�j�h�j�u²����l�y�������������������y�r�l�`�V�[�T�`�l��*�5�6�A�6�6�*�#������������~���������������~�}�~���~�y�~�~�~�~�~�~�3�@�L�Y�Y�e�i�e�Y�L�G�@�3�'�"��'�,�3�3 N  1 & 5 * $ L N 6 V 7  Y g d ) 9 8 * k  G : < # H 0 H T M ] V c  H C = 3 E  E O m S E D � M = Q 6 , m R    ,  0  U  N  Q  �  �  �  o    �  �  x  A  u  �  �  e  �  z  5  �  �  +    Z  �  +  �  A  �  �  �  @  �  �      h  �  R    {  f  !  K  H  �  e  �  r  �  l  @  �<j;o�49X�o��o;�o=t�;�`B$�  <o<#�
<���='�=�P=�-<���=aG�=���<���<�h<�j=�%=��
=L��=�%=�hs=�w=}�=t�=��=�P=#�
=�%=t�=��=m�h>��=�x�=ix�=��=�F=ȴ9=ix�=e`B=H�9=L��=�%=]/>J=�;d=�"�=��-=ȴ9=>	7LB!�B
75B��B�vB
^�B=�B�zB�
B(�>B ��B �B#��B-:B"9Bv�B��BʩB�LB��B ��B#�B%�B�Bd&BڅB��B�_B��BƶB��B�A��9B�B��B~&B��B	+qBa�B
i�B aBb7BqB&*Bu�B߫B�,B�B�B�A�+XB��B+�)B+|B�BLkB"?B
@�B��B��B
v�BBTB��B� B(I�B �5B �WB$=B9lB"BBA�B� B��Bz;B��B �cB"�>B&6�B�B~�B��B=[B>B�B®B��B�7B >�B@tB�B@'B��B	<>BB�B
@�B��B@B�BB�BB��BɯB��B�2B�~A�|TB�cB+�_B�B&QB@@_�AɇwA�7�A�JUA�MgA���A���@���@�(A��A9�vA4E�Au4.A̒A�Y�A���A��aA���A�.oA]�@��@��A���AkyA��(A�+�A�eB�A�&�BH�B
�_A���AA�AY��@g��AÏ�A�u�C��7A��^A��@Ҥ�A<@���@��Ac+�@T	�AÖ;C�k�@ӕ�A��A�G�A��A��!@�?��"@ �A�z2A��vA�d�A���A��A���@�A�@��1A��A9ՌA4�Au�Å~A��mA�z�A���A�kKAɀ�A]-�@���@��A�Aj�A�?A҈�Aܣ�B�"A�WB@4B�A��AAm�AY��@k��A�sA�gC���A���A�u�@��A @���@��PAc��@T2lA�z�C�X@��'A�z�A�z�A�cA���@Ӳ?�_�                  
   )         
         $      N      (   T            ,   <      (   -      !      ?               W      o   K      6   O   :                     K   8   1                                 5                        5   !   !   =            !   5      %   '            +         +      !      #         )   3   #                     '   !   #                                 '                           !      )               3         '                     +                     '   )                              !            M���OJ�NF��N>^�O��N۴4P|�N@��N(�9N��(N�'�OB�OoX�N�ځOC�nO��<O2��P3 jN���O2��M�wWO|jGPVfCO_�N�f�PsxN�H^OO��N�EO�FN��
N�G�PyN��OH/[N��O\�O>@�N�&	P O۪qO���O"ԜN���M�t�N"J)OLCN%��O|��O4҆O�ONy�'N^Z&M�q	N�;  O  D  ?  �  �  �  o  �  *  �  �  A  N    Y  b  ^  �  C  p  n  n  �  �  j  �    1  �  �  �  �     ;  i  �  [        	�  �  	  j  ?  r  �  �  
!  	     �  2  �  
�L�ͼ�j��t��D����`B���
<t�$�  �o:�o;D��;�o<e`B<�C�=]/<49X<�=�w<�t�<���<���<�<ě�<�/=D��<�/<�/=t�<�=L��=o=+=+=C�=�o='�=�hs=P�`=49X=0 �=T��=@�=0 �=<j=<j=<j=H�9=H�9=��w=���=�o=�O�=�v�=�x�=����������������������fgjqt���������tmhgff����������������������������������������fgqstx�����������tgf�������

��������������������������''))466?>;6)''''''''lhkny{}��{unllllllll}���������������������������������������

�������������������������������������������������������2BNY[aa\VIB5)(����������������������������$$�����%')6=BFMB762)%%%%%%����������������������������������������#0<IKUZYUNIA<0#���)/5DGA5��������
#/799/#
�,,/5<AHSTHD<5/,,,,,,��������������������������������������������������������������������������������
	)5BJMONGB5)
JIHKN[cggga^[NJJJJJJgjmtyz�����zvmgggggg #/<AKO\d\UH</##%(++# �������������������������������KGGIMN[gtx����tg[NK��������
!
����roptw����������trrrr�������#/465'
�����������������������������������������ZY`dghot������{th`[Zxtusrtz��������zxxxx$#&)5775+)$$$$$$$$$$HIJO[f`[ZOHHHHHHHHHH������ ��������mnz����������zwnmmmmLJHHJO[hmtx��}th[UOL.,,.5;HMTVYZZWTH;5/.)5BFHJIIHB5)�������������������� #%/;<?<6/#        "),*)&���������	����纤��������������������������������������ÇËÓàãäâàÓÇ�z�s�n�j�h�n�z�~ÇÇ��������������������������������������������������������������������������������������������������������������������������(�����������������������Z�g�s���������������s�Z�N�>�(�$�&�1�A�Z�x�����������������x�r�o�x�x�x�x�x�x�x�x�x�������������x�l�e�l�u�x�x�x�x�x�x�x�x�T�a�m�n�z�|�z�v�m�a�T�N�N�P�T�T�T�T�T�T�A�M�V�Z�[�Z�M�A�A�4�(�$����(�4�6�A�A�������(�4�A�J�A�;�(����������鿒�����ÿѿҿӿпǿĿ�������������������ìù��������ùìãàÓÒËÓàãìììì�/�;�H�T�^�a�g�j�g�a�Z�T�P�H�;�1�'�#�)�/�������*�9�A�D�5�(����ؿƿοѿۿ��A�N�Z�^�g�f�[�Z�N�A�;�5�(�"����(�5�A������/�:�A�C�?�/����������������������zÇÓÔ×ÓËÇ�z�v�n�l�n�u�z�z�z�z�z�z��"�+�.�:�:�A�;�.�(�"��	��������� �	����������������������������������������������������������������r�l�f�d�`�e�r��0�I�UŁňŇ�{�]�0�������������������0�y�����������������y�m�`�H�=�B�G�W�`�m�y���������� ������������������������������)�0�B�O�V�B�:�-���������������āčĖĘĐčăā�t�s�h�[�[�[�h�k�t�{āā�\�h�uƁƏƜƛƚƏƁ�u�h�\�T�O�I�K�O�Q�\�*�6�<�9�6�1�*�������!�*�*�*�*�*�*ƚƧƳ����������������ƳƚƎƉƃƁƅƎƚ�0�=�I�V�Z�_�V�K�I�I�=�0�'�%�0�0�0�0�0�0ŔŠţŭŹźźŹŭŠŗŔŔŒŔŔŔŔŔŔ�M�f��������Ǿ�����s�Z�A�(�!���!�-�M�	�����	���������	�	�	�	�	�	�	�	���!�-�6�:�>�>�:�6�-�!��������������H�J�U�N�H�>�<�/�*�&�*�/�<�F�H�H�H�H�H�H�6�B�O�[�h�k�w�v�m�h�[�O�B�6�4�/�-�.�2�6D�D�D�D�D�D�D�D�D�D�D�D�D�D�D~D{DuD{D~D����	�� �"�)�'�"���	��������������������	��"�1�2�7�<�=�5�/�"����������������4�@�M�Y�f�r������f�P�'����� �#�(�4�ʼּ�������������ּʼ������������ʻ-�:�F�S�_�b�l�u�o�l�_�S�F�:�-� ��!�&�-�S�_�l�x�������������x�l�_�S�M�L�S�S�S�S�.�;�G�K�K�G�;�9�.�+�.�.�.�.�.�.�.�.�.�.������
������������������H�O�U�_�a�c�a�U�H�<�/�#����#�$�/�<�HE�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�Eټ'�4�@�Y�f�n�v�v�r�i�Y�@�4�'�����"�'�����
��#�*�+�#���
������������������¿����������������²¦�p�l�n�y¿�y�����������y�l�f�k�l�s�y�y�y�y�y�y�y�y��*�5�6�A�6�6�*�#������������~���������������~�}�~���~�y�~�~�~�~�~�~�3�@�L�Y�Y�e�i�e�Y�L�G�@�3�'�"��'�,�3�3 N  1 & 5 * ( G a 2 V :  g A d !  8 * k  H 7 ) # H - H & M ] V c  D   ,  E y E O [ S E D � . 5 O U , m R    ,  0  U  N  Q  �  o  a  5  �  �  �  �  �  �  �  t  �  �  z  5  �  �  �  �  Z  �  �  �  v  �  �  �  @  �  �  �  �  �  �  �  �  {    !  K  H  �  �  �  '  �  l  @  �  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  O  H  A  :  4  )       �  �  �  �  �  �  �  �  �  �  �  �  D  <  2  %      �  �  �  �  c  >    �  �    y  �  1    ?  D  J  P  W  ^  e  l  s  |  �  �  �  �  �  �  �  �  �  a  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  x  f  U  C  1  �  �  �  �  w  n  e  `  \  Y  V  R  N  E  0      �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  y  X  /    �  �  �  a  �  �  "  C  X  g  n  k  Z  >    �  �  �  /  �  �  X    �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  s  _  J  C  =         #  %  '  *  +  +  +  +  +  ,  )           �   �  �  �  �  �  �  �  �  �  �  �  �  �  �  v  \  A  $    �  �  �  �  �  �  �  �  �  �  �  l  V  @  '    �  �  �  �  �  k  =  @  >  ;  7  2  +       �  �  �  �  g  =    �  �  �  b  �  
  3  G  M  M  I  >  /    �  �  �  �  [    �  �  B  �  �  �              �  �  �  �  V    �  �    T  �  �    :  L  �    `  �  �    :  P  X  @    �    Q  s  *   �  b  `  Z  P  C  7  &    �  �  �  �    ]  -  �  �  W   �   �  3  l  �  �    5  Q  ^  W  F  ,    �  �  ?  �  C  �  �  '    @  n  �  �  �  �  �  �  w  !  �  >  �  )  �  &  �  �  |  C  ?  :  4  ,  $      	  �  �  �  �  �  �  �  q  ]  I  4  p  j  d  [  R  G  =  2  %      �  �  �  �  �  u  P  )    n  j  f  a  ]  W  O  G  ?  7  ,         �  �  �  �  �  �    B  Z  f  m  m  k  ^  J  ,    �  �  V    �  l  �  z  �  �  �  �  �  �  �  �  �  �  �  �  �  i  7  �  �  K  �  T  �  �  �  �  �  �  �  �  �  V    �  �  �  I    �  r  �  t  �  �  �  �  �  �  �  �    %  <  R  c  j  ]  2  �  �  ?  �  �  �  �  �  �  �  m  6  �  �  [    �  �  �  t  Y  1  �     F        �  �  �  �  �  �  �  �  �  �  �  �  �    )  :  H      +  0  0  +  "    �  �  �  �  X    �  c  �  4  h  o  �  �  �  �  �  }  c  O  >  -    �  �  �  �  d  7  �  �  C  V  #  t  �  �  �  �  �  �  �  �  Y    �  y  �  H  l  ~  �  �  �  �  �  �  �  �  �  �  t  f  X  H  8  '      �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �         �  �  �  �  �  c  =    �  �  �  �  V    �  Z  �   �  ;  0  %        �  �  �  �  �  �  �  �  �  �  �  w  j  ^  
�  0  �  �  =  e  h  Y  D    �  |    
�  	�  	'  �  u  {  �  �  �  �  �  �  �  �  �  �  k  S  3    �  �  8  �  f  �  f  &  �  �  5  O  Z  Y  E  %  �  �  >  �    k  �  
�  �  �  Y  �  �  �    �  �  �  j  .  �  �    k  
�  	�  �  f  �  �  �  �  �  �        �  �  �  �  �  t  L  "  �  �  K  �  �  L    
  �  �  �  �  x  _  H  <  -    �  �  f  �  �  �    �  	a  	z  	�  	�  	�  	f  	  	  	t  	x  	l  	k  	4  �  J  �    �    �  <  �  �  �  �  �  `  4  �  �  w  /  �  �  L  �  I  >  �   �  	    �  �  �  �  �  �  �  �  �  �  l  G  4  9  �  �  Z    H  \  f  X  G  1      �  �  �  �  �  y  Y  4    �  �  �  ?  7  0  (  !                �  �  �  �  �  �  �  }  r  j  b  Z  Q  I  ?  5  +  !    �  �  �  �  r  N  +     �  �  �  �  �  X  L  ?  0  #      �  �  �  7  �  �  H   �   �  �  �  �  �  �  �  p  F       �  �  �  �  g  J  +     �   �  |  	  	m  	�  	�  
  
  
  

  	�  	�  	+  �  /  �  �  "  E  u  j  �    g  �  �  �  �  �  	   �  �  �  ^  �  w  �  K  �  �    �  �    �  �  �  �  �  �  t  4  �  �    �    W  �  �    �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  2      �  �  �  �  �  �  q  ^  L  5      �  �  �  �  {  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  
    �  �  �  �  �  �  �  g  J  (    �  �  v  <  �  a  �