CDF       
      obs    5   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type                     	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?h�t�j~�   max       ?Ѓn��P      �  �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       N�   max       Pb�      �  �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       �t�   max       =�1      �  T   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>��
=p�   max       @FУ�
=q     H   (   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ��p��
<    max       @vm�����     H  (p   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @"         max       @L�           l  0�   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @Α        max       @��          �  1$   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       ;o   max       >�C�      �  1�   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A���   max       B5;      �  2�   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A���   max       B5x      �  3�   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       >��%   max       C��      �  4t   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       >���   max       C��      �  5H   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max               �  6   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          7      �  6�   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          '      �  7�   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       N�   max       P�Y      �  8�   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�>�6z�   max       ?�J�L�_      �  9l   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       �t�   max       >bN      �  :@   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>��G�{   max       @FУ�
=q     H  ;   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ��(�`    max       @vi\(�     H  C\   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @(         max       @L�           l  K�   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @Α        max       @�@          �  L   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         ?�   max         ?�      �  L�   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�e+��a   max       ?�I�^5?}     �  M�                                 (      '         /        
         
            K         ?                           &      �                      i   6   V      :   3   
NRwN�O@��O��ONkN�V7PhlO�n�O��NL �O�4�O7]�P;\�N�aN� P*DDN�(yPTvO\��O]�oN,ivNc-�NO�~O��N��Pb�N�SfN��PC�OG��O?n�Ok�NbN�WN2G(O'�6O
�O�?�O!^5Oƥ�N.f]N��N�fLO��	O+70OP�O�V�PX�O�8&N�O���O�N����t��o:�o;o;D��;ě�<o<#�
<#�
<#�
<#�
<49X<49X<49X<49X<49X<D��<u<�C�<��
<�1<�9X<�j<�j<���<���<���<���<�/<�/<�/<�<�=o=C�=\)=�P=�w='�=,1=,1=,1=H�9=L��=L��=T��=]/=q��=�o=�C�=�\)=���=�1����������������������������������������.019<AIOUbehhebUF<0.�������������r{|{z{�������{rrrrrr)1586752)"kimz�������������vqk�������������������������	
����������������������������������

������"/0;HJNLJH;/)"����� )2,%����������������������������).)) ������&=BNTg������tg[H<5-&�"%%!������@EP[gt���������g[JD@wusrz������������{zwidbbmyz����������zmi����������������������������������������LKN[ging[NLLLLLLLLLL��������������������)5665)�����)451%�������mootw����������tmmmm���������������������
#<HWZ`ilaU/
�/<@BAHUbknnhcaUQH86/�����	 !&����MOUanz���������zn]UM����������������������
#%%($#
�����������	������������569<HP_aaba_WTHFDD<5���������������������������%"&%!��eefghkst��������tkhe������

������������������������������
"#$##
������ZZ[ehtxxwvth\[ZZZZZZ)BDHIGA5)++57BN[^cd`[PNB@?75+������
#%)(!
����uz���������������~zu��������
	�������������� ).43/)#
���������������������95/346;HTaiopla[TH;9}������������������}')6@=763)"��������������������������������*�6�=�C�C�C�6�*�&�������������������������ûɻ����������|�t�x�|���������	��������	�����������������𼋼���������������������������������������"�%�.�1�2�.�'�"��	����������	����s���������������������g�A�5�,�*�3�N�Z�s�������������������������������������������ûлۻڻ׻ͻû����������}��������������������������ݻ���������A�M�S�Z�\�Z�V�F�4�(���������� �	��(�A�T�a�i�m�t�z�z��t�m�a�H�;�8�;�E�H�L�M�T���*�N�Z�d�e�Z�J�A������ؿпϿڿ��Ƴ����������������ƳƧƞƚƗƚơƧƫƳƳ�h�u�~�}�|�y�|�u�h�h�\�Z�Y�Y�\�e�h�h�h�h�5�N�X�V�[�b�c�[�B�5������������"�)�5�z�|�������������z�m�i�m�r�y�z�z�z�z�z�z��)�B�O�W�[�_�W�O�6�������������������5�A�N�Z�e�`�Z�X�R�N�A�5�(�#����(�*�5�������������������������������ƼƸ�����/�<�H�N�H�F�<�7�/�-�)�+�/�/�/�/�/�/�/�/�m�z���������z�m�k�i�m�m�m�m�m�m�m�m�m�m�������������������������������������������ʾ�������߾׾ʾ������������������f�f�j�n�n�k�f�Z�X�N�M�Q�Z�f�f�f�f�f�f�f�/�;�T�m�u�s�c�H�"�	���������������"�/����!�(�3�4�6�4�(�����������������������������y�v�~����������'�A�s����������f�M�4�(�������	���/�/�;�H�T�Y�U�H�;�/�"��	���	���"�/�:�F�S�a�l�t�l�b�S�:�-�!������!�.�:���������������������������������������������������������������������������������������	�
�������߼���������#�/�3�0�/�#���
����������������	������������ùììöù��������(�.�4�7�A�E�F�A�4�(��������� �(��"�.�;�G�T�d�h�g�`�G�;�"��	�������������������ĽнӽѽĽ����������~�y������D�D�D�D�D�D�D�D�D�D�DoDQD@DIDVDbDoD{D�D������������������������������������������m�n�s�v�m�l�`�T�Q�G�B�G�J�T�`�d�m�m�m�m�-�:�;�F�L�N�F�:�-�!���!�!�-�-�-�-�-�-��#�0�<�;�6�*��
����������������������b�n�z�{�|Ł��{�n�b�U�I�<�:�6�<�I�U�W�b�����������ĺƺ������������~�r�l�q���������������������Ϲù����������Ϲ���<�H�P�Z�[�T�I�8�#��
����������������<EuE�E�E�E�E�E�E�E�E�E�E�E�E�ErEiEfEfEiEuìù��������ùõìåìììììììììì�hāĚĿ�������������ĿĦċā�r�^�c�h���¼ʼӼݼּܼмʼ���������i�s����������������������������~�y�~�������������� G Y ' H C " ` B . \ G 5 ' * V 1 B  ;  L ! 4 H _ J ` $ 5 0 a Z r E A W e ' D ? P 4 ? . 2 # % A 4 L n A >    m  T  �    �       3  �  q  h  �    �    �  �  Z  �  �  c  k  b  =    �  �  �  J  �  �    �  �  9  �  �  }  a  �  e  �  �  �  u  �    p  �  I  �  #  �;o;o<��
<�9X<o<T��=��<�1=t�<�o=L��<�`B=L��<�o<u=m�h<�o>�C�<���=H�9<���<��<�/<�<��=���<�=t�=�Q�=\)=@�=,1=\)=�P='�=ix�=H�9=���=q��>A�7=8Q�=q��=q��=��=�o=��->"��=�l�>��=���>o>   =��B�B�1B&�5B�ZB)6{B�iB ��B�B"�{B"9#B#��A���B�B�B��B	�5B�`B
DBR�B "MB +&B�9B�6B5;BK�BYEB�uB#P�B4]B��B�B(pBP�B$��B�BA�B��BȯBwMBsB�ZB�'B�B��B�B$:BJ�BؠBݸBGQA��qB�,BWtB��B��B'>SBI	B)��B>�B C�BF�B#�SB"?�B#_�A���B=1B��B��B	��BB	��B�3B :�B $�BԉB�zB5xBD�B0�BB�B#]^B?�B-�B�VBG�B�%B$�0B:B?B=�B��B:B>gB tB4B�kBRBA�B#�BEJB�B�%B@�A���B��BA4A��A��@���A��@��A]*A���A�v>@��@��A73�A�o�A��bBSoB�^A�^A�A�x�A��B�A�*aA���AI�sAR/A?��A�h%A4{P@��EA;�-A�}�@}�'A�TA��A�SA�^�A��@A7SAabmA"N
C��2At��Ah��@wx&A��A�<@��>��%A�tC��A�u�A�n@�P�@Q�A��JA���@�W�A��B@���A]�A���A��@��8@�2�A7A��A�f�BA{BBOA�.�A��A�~�A��@B��A��A��AIۘAR��A?JA�~A17@���A>��A��@|SA���A�{AúA�|7A�u�A8��Aa1�A#�[C��1At�^Ah�@vA�wA�o�@:�>���A��C��A�}�A�@���@@�                                 )      (         0        
         
            K         @   	                        &      �            !         j   7   W   	   ;   3                        '                  +         -      -                        7         -                                 !            !         #   %   !      '                           '                  !                                       '         #                                                                        NRwN�O$�/O?p�NkN�V7PcO�n�O,?NL �O���N�ߺO͹N�aN� O�}XN�(yO��FO\��O*k�N,ivNc-�NO�~O��Nv�O�ԂN�SfN9|�P�YO%�^O1W�Ok�NbN�WN2G(Ny��O
�O�\�O!^5O+�uN.f]N��N�fLO�NO�OP�O���O���OR�nN�O��(OZ��N���  �  �  �    �  �  �  +  �    �  �  �  �  �  �    �  �  �    [  �  &  �  �  �  �  m  X    �  �  �  \  ~  5  O    =     &  �      Y  �  �  0  I  
U  	L  Ҽt��o;D��;ě�;D��;ě�<#�
<#�
<�C�<#�
<T��<�t�<ě�<49X<49X<�/<D��>bN<�C�<���<�1<�9X<�j<�j<���=@�<���<�h=#�
<�`B<�`B<�<�=o=C�='�=�P=8Q�='�=���=,1=,1=H�9=Y�=P�`=T��=�1=���=ě�=�C�=�9X=���=�1����������������������������������������624<CISUXbfgedbYUI<6�������������r{|{z{�������{rrrrrr)1586752)"mkmz�������������xrm������������������������������������������������������������
�������!"/;?EC;/)"�������" ���������������������������).)) ������NJLNRYgt�������tg[TN�"%%!������[XY\bgt����������tg[wusrz������������{zwigimz�����������zyni����������������������������������������LKN[ging[NLLLLLLLLLL��������������������	)44)										�����)./-)�����mootw����������tmmmm����� ������������		#<HPRS[\YH/	;9<>CDDHSUaillgaUH<;������ " ���MOUanz���������zn]UM����������������������
#%%($#
�����������	������������89<BHIUXZUSHD<888888���������������������������!"���eefghkst��������tkhe�������	

����������������������������
"#$##
������ZZ[ehtxxwvth\[ZZZZZZ		
,8@EGE>5)	--5@BN[]bc_[YNBA95--������
#%)(!
���������������������������������������������
#()(#
������������������������::;AHTXadiigaTHA>=<:��������������������')6@=763)"��������������������������������*�6�=�C�C�C�6�*�&�����������������������ûŻû�����������x�v�z����������	�����	���������������������𼋼���������������������������������������"�%�.�1�2�.�'�"��	����������	����s���������������������s�N�5�/�-�6�N�Z�s�����������������������������������������������ûлллͻŻû���������������������������������ݻ���������4�A�M�Y�X�S�C�4�+������������(�4�a�m�n�r�q�m�a�T�K�P�T�X�a�a�a�a�a�a�a�a���(�5�;�J�T�V�U�N�A�5����������Ƴ����������������ƳƧƞƚƗƚơƧƫƳƳ�h�u�~�}�|�y�|�u�h�h�\�Z�Y�Y�\�e�h�h�h�h���)�-�5�B�O�T�U�N�B�5������������z�|�������������z�m�i�m�r�y�z�z�z�z�z�z����)�6�=�F�H�F�<�6�)�������������5�A�N�Z�e�`�Z�X�R�N�A�5�(�#����(�*�5��������������������������������ƾ�������/�<�H�N�H�F�<�7�/�-�)�+�/�/�/�/�/�/�/�/�m�z���������z�m�k�i�m�m�m�m�m�m�m�m�m�m�������������������������������������������ʾ�������߾׾ʾ������������������Z�f�h�l�l�f�Z�P�O�S�Z�Z�Z�Z�Z�Z�Z�Z�Z�Z�;�H�T�`�h�e�a�Z�M�;�"�������	��"�0�;����!�(�3�4�6�4�(����������������������������|�z��������������������(�4�f�s��������u�f�Z�A�(�������(��"�)�/�;�H�T�W�T�T�H�;�/�"��
�����:�F�S�^�_�l�r�`�S�:�-�!������!�0�:���������������������������������������������������������������������������������������	�
�������߼���������#�/�3�0�/�#���
����������������������������������������������(�.�4�7�A�E�F�A�4�(��������� �(�"�.�;�G�O�T�^�a�`�T�G�;�"��	��� ���"�����������ĽнӽѽĽ����������~�y������D�D�D�D�D�D�D�D�D�D�D�D�DtDoDeDoDsD{D�D������������������������������������������m�n�s�v�m�l�`�T�Q�G�B�G�J�T�`�d�m�m�m�m�-�:�;�F�L�N�F�:�-�!���!�!�-�-�-�-�-�-��������#�0�8�8�0�#��
�����������������b�n�y�{�{ŀ�~�{�n�b�U�K�I�=�A�I�U�Y�b�b�����������ĺƺ������������~�r�l�q���������ùϹܹ�����������ܹϹù������������#�/�<�H�K�R�K�>�-�#��
�����������
��#E�E�E�E�E�E�E�E�E�E�E�EzEuErErEuE{E�E�E�ìù��������ùõìåìììììììììì������������������ĿĳġđčĚĦĳ���ؼ����ʼѼۼڼּ̼�������������������������������������������~�y�~�������������� G Y & G C " ` B % \ ? E $ * V + B  ;  L ! 4 H W H ` 2 2 0 _ Z r E A ! e " D - P 4 ? 4 , # " J ' L ` - >    m  T  f  �  �     �  3  i  q    �  �  �    X  �  $  �  m  c  k  b  =  �  _  �  Y  [  m  �    �  �  9  z  �    a  p  e  �  �  n  G  �  #  K  �  I  R  �  �  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  �     .  G  }  �  �  �  �     	      #  *  2  C  ]  w  �  �  �  �  �  �  �  �  �  {  k  Z  G  4       �  �  �  �  �  �  �  �  �  �  �  �  �  n  Q  ,    �  �  ~  X  <  1    �  �  �  �        �  �  �  �  �  h  7    �  {  7    �  i  �                            �  �  �  �  �  �  �  �  �  �  �  �  �  }  i  H  #     �  �  �  s  N  (     �  �  �  �  �  �  �  �  �  �  �  `  6    �  �  ;  �  �  -  �  +  $         �  �  �  �  �  �  t  [  E  .    �  �  �  H  �  �  �  �  �  �  �  �  �  y  j  [  1  �  �  P    �  �  �    �  �  �  �  �  �  �  �  �  �  �  �  �  �  ~  �  �  �    �  �  �  �  �  �  �  �  W    �  �  N  �  �  5  �  Q  �  [  �  �  �  �  �  �  �  �  �  �  �  �  p  5  �  �  T  �  m   �    >  Z  n  }  �  �  �  �  m  J    �  �  �  �  ?  �  �  )  �  �  �  �  �  �  p  `  P  >  ,        
    �  �  �  �  �  �  �  �  ~  w  r  n  j  e  [  L  <  ,      �  �  �  �  �  �    O  q  �  �  �  �  |  e  G  #  �  �  p    �  ?  �        
       �  �  �  �  �  �  �  �  �  �  �  �  �  �  [    �    2  -  �  g  �  �  �  c  �  �  �  "  &  \    �  �  �  �  �  �  }  f  N  6  *  '      �  �  �  �  ~  5   �  r  �  �  �  �  �  r  Z  :    �  �  }  0  �  l  �  ~  "  +      
         �  �  �  �  �  �  �  �  �  �  �  �  �  �  [  Q  G  <  0  $      �  �  �  �  �  u  M  #  �  �    A  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  ~  q  d  W  &      �  �  �  �  �  �  �  �  �  p  _  J  0     �   �   �  }    �  �  ~  w  p  g  \  Q  D  3  #    �  �  �  o  .   �  �  +  B  h  �  �  �  �  �  b  0  �  �  R     �  @  l  _  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  y  t  p  m  i  e  �  �  �  �  �  �  �  �  �  �  �  �  �  �  z  B    �  I  �  �  1  Z  j  m  h  X  >       �  �  o  %  �  H  m  �  �  �  L  Q  V  R  H  <  ,      �  �  �  �  �  �  �  �  �  �  �           �  �  �  �  _  5  	  �  �  x  C  �  �    �  >  �  �  �  �  �  �  �  �  w  P  1    !  #        �  �  q  �  �  �  �  �  �  �    o  ]  K  9  $    �  �  �  �  �  |  �  �  �  �  ~  o  a  S  D  6  (           �  �  �  �  �  \  `  e  f  b  ^  W  O  G  :  -    
  �  �  �  �  b  3    �  �  �  �  �  I  v  m  d  Q  :  !     �  9  �  �  @  �  �  5  +  !      �  �  �  �  �  u  L  W  [  I  ;  /  !      0  ?  K  O  N  I  >  .      �  �  �  �  P  	  �    ~          �  �  �  �  �  q  E    �  �  U    �  �  i    �  �  �  �  �  �  �    ;  8  ,     �    .  1  !  �  w  �  
           �  �  �  �  �  �  �  �  �  �  �  p  \  I  5  !  &  "          �  �  �  �  �  r  K  !  �  �  �  p  G    �  y  r  j  a  R  ?  '    �  �  �  |  H    �  �  r  @            	  �  �  �  �  �  m  8  �  �  @  �  J  �  Z  �        �  �  �  �  �  v  V  ;  $  
  �  �  y  %  �  a   �  Y  L  ?  0       �  �  �  �  `  .  �  �  �  C  �  �     �  �  A  �  �  �  �  �  �  �  M  �  e  �    
9  	+  �  z  �  x  q  �  �  �  �  �  �  �  �  �  b  !  �  a  �  D  �        	�  
�    q  �  �  "  0  "  �  �  t    
�  	�  	0  �  |  J  >  I  :  *      �  �  �  [  $  �  �  q  -  �  �  [     �   |  	�  	�  	�  	�  
  
E  
T  
H  
#  	�  	�  	S  	  �    u  �  �  5  ;  �  	4  	J  	5  	  �  �  �  <  �  �  S    �  >  �  	  �  �  L  �  �  �  �  y  `  H  2      �  �  �  �  s  R  -    �  �