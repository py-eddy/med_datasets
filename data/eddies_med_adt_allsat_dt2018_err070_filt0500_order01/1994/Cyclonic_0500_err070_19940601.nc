CDF       
      obs    O   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type         ����         	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?`bM���   max       ?�\(��     <  �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       N�]   max       P��i     <  �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       ���   max       <�j     <   $   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?aG�z�   max       @F!G�z�     X  !`   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ��(�\    max       @v|(�\     X  -�   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @0         max       @Q�           �  :   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @ˣ        max       @���         <  :�   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       �"��   max       <�C�     <  ;�   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A�H   max       B0J�     <  =(   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A���   max       B0K�     <  >d   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       =0�   max       C��Y     <  ?�   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       >�w   max       C��q     <  @�   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          �     <  B   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          A     <  CT   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          1     <  D�   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       N�]   max       PiO     <  E�   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�:)�y��   max       ?������     <  G   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       ��
=   max       <�j     <  HD   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?fffffg   max       @F!G�z�     X  I�   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ���G�|    max       @v|(�\     X  U�   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @(         max       @Q�           �  b0   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @ˣ        max       @��         <  b�   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         ?^   max         ?^     <  d   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?������   max       ?������     �  eH            �                  
                        $                                    &                              
   :                           ?   +      !      	   (         9      
                     I                     	            8NI�<N_yO�q�P��iOkt�N$q�N�e�N2�N���OɀN)�	O ��N�F=N��O��3N�FNW�LP+l7Nof�N�}N���N]�TO4�oO�6&N�% O�O���O9}CN�"�Pe�yNu�P#��Oj��O�R�Nd�
NhxOn��N�]Ol��NǺ<P:��O�
�N�$�N >�Nw�%N��O��qN�3#N�e�OࣥOY2�N��O�˼O��N�O���OA�N��P�N�7�N��N�}8N�tN�O�YFN0�N2�O��O���O�N��BN��BN{�N��N�O_O�y�N�1HN9&O&��<�j<���<�C�<�C�<�o<e`B<T��;ě�;ě�;�o;o�o�o�o�D�����
���
��`B�o�#�
�49X�D���T���u��o��C���C���C���t���t���t����㼣�
���
���
���
��9X��9X��j��j�ě��ě��ě��ě��ě����ͼ�`B��h���o�o��P��P��P��w��w�''0 Ž0 Ž49X�49X�8Q�8Q�D���P�`�T���Y��]/�m�h�m�h�y�#�y�#�}󶽑hs��hs��t�������
!
����������������������
/:9/&#
�����������*BW[XO6�������������������������#$/:5/%#inyz������zncciiiiiiz��������|zzzzzzzzzz36BOXX[[\[UOBB:62/33#0;A:5) �������������������������� 
 ������������������~��������~~~~~~~~~~#/<HUW_UPH</'#����������������������������������������)@B`hpst~t[O	 ),/)PT_aimmmmhaTLLPPPPPP��� �������(*,36ACDJIC?6,*$((((��������������������ABF[gt���}sg[NMFB@>A��������������������������������������7B<5)
��������������������������%)6;;=6/)g����������������tfg������������������������������{xx�LN[gt��������tr[WJIL��������������������&)/6BOTOMB=60)&&&&&&]hlt�����thc]]]]]]]]������������������rt|�������ztrrrrrrrr��������������������������������������Ohwywh[O6( ��6OEHTamz�����~zmTHDBCE~���������������~~~<<HTUXXUHHB<<<<<<<<<Xanz��znnbaUXXXXXXXX��������������������U[g������������to_OU�������������������������������������	#0GRVTO<10#
���"$*06<IQUYYWMI<0)%#"��������������������8<HUaz������znZUSD78������������������wz�����}zqwwwwwwwwww)26;CO[h|}ywthOB6)()/5;BNP[glpmg[NB>51//������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������qrz������������������egt}��������tgfaadeeno{��������{vqomrmnxz��������������zwxx��������������������OU`adjaUSQOOOOOOOOOO���������������������������	��������������������������;<HMUVUNH><9;<<<;;;;��
#*('$#
������������������żƼ������������������������l�b�`�_�`�l�l�l�y�|��y�l�l�l�l�l�l�l�lÜ×Ù×Üàì����������������������ìÜ�����������û��8�Y�e�����������f�M����
���������
��#�/�3�<�K�S�H�<�/�#��I�A�=�1�0�-�0�=�A�I�K�K�I�I�I�I�I�I�I�I�ܻػлϻλлܻ������������ܻܻܻܻܻ����������������������������������������һS�O�I�I�S�_�l�r�x�~���������x�r�l�_�S�S���$�)�0�4�=�I�V�b�o�q�g�b�_�V�R�0�$��A�=�4�(� ��(�4�5�>�A�H�A�A�A�A�A�A�A�A�y¦²¾¿��¿¼²¦�B�@�5�4�5�@�B�K�N�[�g�q�g�[�X�N�B�B�B�B�Ľ��½Ľнݽ�ݽٽнĽĽĽĽĽĽĽĽĽ��O�6�)�'�'�*�'�&�)�B�M�[�h�t�v�t�p�j�[�O�������������������������������������������������������������������������������׿y�m�`�G�B�.�&�.�G�`�y���������ɿĿ����y�m�c�i�m�v�y���������z�y�m�m�m�m�m�m�m�m�����������������������������������������a�W�U�R�T�U�a�m�n�n�p�s�zÃ�z�n�a�a�a�a���	����������	����"�#�"������������������������������������������������������������������!�����������b�]�U�I�D�D�I�U�b�n�{�{Ńņ�{�n�b�b�b�b�"����	��	��"�#�.�;�;�F�K�G�B�;�.�"�������~�������������þ׾���׾ʾ������U�K�H�?�<�/�+�<�U�a�n�q�s�z�|�z�n�f�Z�U�f�e�Z�S�O�Z�f�s�w�������s�f�f�f�f�f�f�����f�A�(�����4�M�s������ھ� �������;�5�4�8�;�?�G�N�K�G�;�;�;�;�;�;�;�;�;�;�����l�[�R�T�y�����Ŀֿ�������ݿĿ������������	��"�$�&�#�"���	��������ھ۾����	��"�+�.�3�,�"���	���/�&�#����#�*�/�9�<�?�<�7�/�/�/�/�/�/�/�'�#�� �#�/�;�<�F�?�<�/�/�/�/�/�/�/�/�)�%�(�/�1�8�B�O�[�h�xĂā�t�h�[�O�B�6�)�;�6�/�,�/�;�E�H�H�P�H�>�;�;�;�;�;�;�;�;������������������������� ��������/�#�"���	���	��"�$�/�9�:�;�?�;�/�/�g�t�������	���������������s�f�c�l�o�g�(�!�!�����(�5�>�A�N�T�^�j�i�Z�N�5�(�g�a�Z�N�K�N�O�Z�a�g�s�s�����������s�g�g�#�#��#�#�/�<�H�<�<�/�#�#�#�#�#�#�#�#�#�H�G�F�E�H�U�]�a�b�a�a�U�H�H�H�H�H�H�H�H������������(�5�7�7�5�(�'��������a�`�c�c�g�g�l�l�m�z�����������������m�a������żŹŹŹ���������������������������N�K�C�J�N�Z�g�s�������z�s�l�g�]�Z�T�N�N�������w�n�j�y���������ѽݽ��ݽ̽Ľ����������������(�4�A�M�Q�T�O�A�4�(�ƧƣƥƧƳ��������ƳƧƧƧƧƧƧƧƧƧƧ�����~�v�t���������ʾ׾�޾�����׾����l�_�S�:�-�������!�-�F�Y�l�������x�lE�E�E�E�E�E�FE�E�E�E�E�E�E�E�E�E�E�E�E��ܹعϹù������ùϹܹ�� ��	�� ��������������$�*�6�C�J�N�N�F�C�<�6�*��V�K�I�=�2�=�E�I�V�`�b�l�o�r�o�b�V�V�V�V���p�e�`�c�s�����ֺ���� ���⺽��������	�����!�-�:�F�K�S�Y�S�F�:�-�!�����������������ɺֺ����ֺɺ����������~�w�r�j�e�c�e�k�r�~�������������~�~�~�~�r�q�g�q�r���������������������r�r�r�r�������üʼּټۼּʼ������������������������������������������0�6�C�B�<�#��
��������������������	�������������������r�r�r�x�����������������������r�r�r�r�������6�O�[�f�i�e�[�B�6�����������ּͼƼż˼Ҽ޼�������������������¿¿¿¿�������������������������˻����������4�@�F�M�P�F�@�4�'����������������������������ùƹϹѹϹ̹ù���FFFFF$F1F3F8F1F$FFFFFFFFFF�������������������������������������������������� �!�$�'�,�(�"�!�����l�S�G�;�3�:�<�B�S�y�����������������~�l����������������������������������������EuEtEoEuE�E�E�E�E�E�E�E�E�E�E�E�EuEuEuEuEEEEEEE
EEE*ECEEENEHECE7E5E*EE - N 4 ? Z m 7 A + � � 6 K T 4 Z W I ^ ; O c ; 3 Q * 6 8 P g a O X   V @ G ; 9 V R 6 , b g S t T A , ; < $ o O 4 5 F > f s X 9 H ; b m F S 4 d P 6 Z y < D c -    N  V  7  �    p  �  �  
  �  {  "  �  I  D  �  l  &  �  �  �  �  �    �  *  �  �  �  i  _  /    �  �  �  �  .  �  �  �  '  	  c  �    �    �     �  >  C    )  c  �  �  �  '    �  �    �  \  �  �  �  J  Z  �  3  ?  E  �  �  j  h<�C�<�o�ě�����`B<D����o��o�D���ě��o�u��`B�o�o�o�#�
�0 żD����t���C���o�ě��0 ż�9X��h�49X�0 ż��ͽe`B��1�<j�'0 ż��ͼ��ͽ����ͽt��+���T�#�
�o��`B�\)�8Q�#�
�,1�49X�\�����,1��\)�e`B�D�����w�]/�T�������Y��]/�L�ͽ�o�H�9��%�ixս]/�   ��1��\)�����O߽�O߽�+���
��E�������"��B$��B+4B�B��BLB·B>�B��Bq�Bm�BO�B�5B@�B)_�BLBn�B��BA)B4�A�HB�[B0J�B�vB��B�lBv"B�~B �xBR�B��B.�B*d?B	��Bg�B�BB ��B�NBݶB!5�B�}A��!B ץBI�B�B4�B
��BV|B��B%��B&>CBj�BL�B"}�B��B��BE�B,�Bi�B�-B �B!�}B�#B��B"�B�BB+�Bx�B-4�B	�6B) B��By�B�`B|%B)�BpaB�B�B$D�B+�B�B8�B��B��BA�B�BG�B@�B��BP(Bd�B)�'B=�Bh�B�4BB�BM�A��tB�1B0K�B��B	:�B��B�!B@B �YB9B��B.�B*�B	�9B��B�lB1�BG�B�"B�}B!AB�`A���B>oB�VB��BA�B
�ZB;�B­B%�?B&@�B�AB��B"@]BA�B?�B=�B@BG�BCmB ��B"H�B�;B�IBB��B+A.BEB-�gB	�gB(�9B?!B��BåBCLB?B~�B?�BČ@�-�AI<A͵�@�!�A��fB
��@��A�m�@�|�B
��A8��A�wA��
A)�A�47A��A���Ao�@AmhB;FA�5cA[�A�a^A�m�A�ĉA`~DAL��Aś�AB#AJITAc�.Aw��AZ��AZk�A��A'A٭A�öA�ɧA�5�A�L|A�@$A��A�A��A���A���A�#]A�/�A �A6��B8;AL�2@��"C���>�0�A��B�L@%G@tX<@3:�@�@�z�@��A���A�q�@���A�,_A��A�K[@��=0�C��YAt��A
��AR�A��~C� #C��@��AیA͋�@��A�wTB
ŝ@��TA�*@��B@�A6�hA�{�A��PA*�wA؉�A�B�A���Ao/�An��BL�A�6A[A�18A���A�PAa(AJ��A�AB�AN�|Ac��A{��AZ�dAYpAACA©4A�M�A�`$A���A��GA�S�A�| A���A�s�AŀA��<A��SA�{�A�g�A lA7
dBBAJ�@l#C���>�wB @zB��@(B@mʞ@2�M?��c@�@@���A�vAэ�@���A�|HA�A��w@�!�C��qC���Asm�A
�A7�A�|EC��C��            �                                          %      	                              '                                 :      	                     @   ,      "      
   (         9                           I            	   	      
            9            ?                                          -                           #         A      3                           3                           '         )   '               )                           $                        !                     1                                                                              /      /                           !                                    !   '               )                                                   !         NI�<N_yOP��PiOONFN$q�N�νN#�N���OɀN)�	O ��N�F=N��O�l�N�FNW�LO���Nof�N�}Nu8�N]�TO4�oOB��N�% N�ǲO�֛O��N�"�P �Nu�P��O7�O=�TNd�
NhxO[�&N�]O _N �O���OPyN�$�N >�Nw�%N��O��qN�3#NP;�O0O3�gN��O��iO��N�O[B�OA�N��P�N�7�N�L�N�}8N� �N�O�YFN0�N2�O_m�Or�!O�N�CN� �N{�N��N�"�O�y�N�1HN9&O3  8    /  �  �  �  |  �  �  �  �  �  ]  �  �    J    S  B  �  �    �  e  1    �  �  #  �  �  �  �  Z  U  R  d     �  �  �  �  t  �  �  )    �  \  �  �  "  Y  �  �  �  �  �  �  P  �  N  �  �  '  o    �    1    �    �  p       <�j<���<49X��j<e`B<e`B<D��;�o;ě�;�o;o�o�o�o�ě����
���
��1�o�#�
�D���D���T�����
��o��t����
���
��t����ͼ�t����
��j���ͼ��
���
��j��9X��/��/�#�
��/�ě��ě��ě����ͼ�`B��h�C��y�#��P��P�,1��P��w�H�9�''@��0 Ž8Q�49X�@��8Q�D���P�`�T�����P�m�h�m�h�q���}�y�#�}󶽓t���hs��t������
=
!
�����������������������
#%/0/(
����������&8:6)��������������������������#$/:5/%#jnz{������znfdjjjjjj��������36BOXX[[\[UOBB:62/33#0;A:5) �������������������������� 
 ������������������~��������~~~~~~~~~~#/<HTZUMF<2-#	����������������������������������������!-6BO[hpoh`[OB) ),/)PT_aimmmmhaTLLPPPPPP�������������(*,36ACDJIC?6,*$((((��������������������ABIN[gtz���yng[NKFDA���������������������������������������.66'��������������������������%)6;;=6/)r��������������xtkkr��������������y����������������{xyMNS[gt�������tg\[NNM��������������������&)/6BOTOMB=60)&&&&&&]hlt�����thc]]]]]]]]��������������������rt|�������ztrrrrrrrr����������������������������������������#)6BO[hmpje[O?96-!#EHIamz����|zsgTHFDDE~���������������~~~<<HTUXXUHHB<<<<<<<<<Xanz��znnbaUXXXXXXXX��������������������U[g������������to_OU���������������������������   ����������#-02<@DFC<;0#+0<?INUWVTI<40+'%$&+��������������������;@HUanz�����zaUHB:9;������������������wz�����}zqwwwwwwwwww0:@IOX[hutqrqh[OB6/0/5;BNP[glpmg[NB>51//�����������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������egt}��������tgfaadeeu{|����������{wsqpvuzz����������zxzzzzzz��������������������OU`adjaUSQOOOOOOOOOO���������������������������	��������������������������;<HMUVUNH><9;<<<;;;;��
#('&%##
������������������żƼ������������������������l�b�`�_�`�l�l�l�y�|��y�l�l�l�l�l�l�l�lìäáßÝßàëìù����������������ùì�4���׻ɻŻ˻ܼ�@�Y�g�r�{��~�{�f�M�4���
���������
��#�/�2�<�I�Q�H�/�#��I�A�=�1�0�-�0�=�A�I�K�K�I�I�I�I�I�I�I�I�ܻٻллϻлܻ������������ܻܻܻܻܻ����������������������������������������һS�O�I�I�S�_�l�r�x�~���������x�r�l�_�S�S���$�)�0�4�=�I�V�b�o�q�g�b�_�V�R�0�$��A�=�4�(� ��(�4�5�>�A�H�A�A�A�A�A�A�A�A�y¦²¾¿��¿¼²¦�B�@�5�4�5�@�B�K�N�[�g�q�g�[�X�N�B�B�B�B�Ľ��½Ľнݽ�ݽٽнĽĽĽĽĽĽĽĽĽ��O�6�0�)�(�)�,�*�6�B�G�[�h�r�t�t�i�f�[�O�������������������������������������������������������������������������������׿����m�e�]�Z�Z�e�y�����������������������m�c�i�m�v�y���������z�y�m�m�m�m�m�m�m�m�����������������������������������������a�Z�U�S�U�X�a�n�n�p�zÁ�z�n�a�a�a�a�a�a���	����������	����"�#�"������������������������������������������������������������������������������b�]�U�I�D�D�I�U�b�n�{�{Ńņ�{�n�b�b�b�b�"������"�'�.�7�;�D�G�I�G�;�9�.�"�"�����������������������ʾ׾�߾׾ʾ������U�Q�H�A�<�.�<�C�U�a�f�n�p�q�z�z�n�c�W�U�f�e�Z�S�O�Z�f�s�w�������s�f�f�f�f�f�f�����f�M�>�M�Z�s�������о������侾�����;�5�4�8�;�?�G�N�K�G�;�;�;�;�;�;�;�;�;�;�Ŀ����n�\�V�m�����Ŀѿݿ�������ݿľ������������	���"� ����	�	���	�������������	��"�%�+�%�"��	�/�&�#����#�*�/�9�<�?�<�7�/�/�/�/�/�/�/�'�#�� �#�/�;�<�F�?�<�/�/�/�/�/�/�/�/�)�(�)�/�2�9�B�O�[�h�wĀ�}�t�h�[�O�B�6�)�;�6�/�,�/�;�E�H�H�P�H�>�;�;�;�;�;�;�;�;�����������������������������������
����"�+�/�/�/�"�����������}�}�w�u�y�����������������������������5�*�(�!��%�%�(�5�8�A�F�N�Z�e�b�Z�N�A�5�g�a�Z�N�K�N�O�Z�a�g�s�s�����������s�g�g�#�#��#�#�/�<�H�<�<�/�#�#�#�#�#�#�#�#�#�H�G�F�E�H�U�]�a�b�a�a�U�H�H�H�H�H�H�H�H������������(�5�7�7�5�(�'��������a�`�c�c�g�g�l�l�m�z�����������������m�a������żŹŹŹ���������������������������g�f�Z�N�L�N�S�Z�g�s�z�t�s�g�g�g�g�g�g�g�������~�y�x�y���������������½½����������� ��	���(�4�A�M�Q�M�K�A�4�(���ƧƣƥƧƳ��������ƳƧƧƧƧƧƧƧƧƧƧ�������z�x�����������ʾԾԾ��ؾʾ������l�_�S�:�-�������!�-�F�Y�l�������x�lE�E�E�E�E�E�FE�E�E�E�E�E�E�E�E�E�E�E�E��ܹϹù��������ùϹܹ������ ����������������$�*�6�C�J�N�N�F�C�<�6�*��V�K�I�=�2�=�E�I�V�`�b�l�o�r�o�b�V�V�V�V�����s�h�c�g�x�����ֺ��������ຽ������	�����!�-�:�F�K�S�Y�S�F�:�-�!�������������ɺֺ����ֺɺ��������������~�w�r�j�e�c�e�k�r�~�������������~�~�~�~��t�r�j�r�w������������������������������üʼּټۼּʼ������������������������������������������0�6�C�B�<�#��
��������������������	�������������������r�r�r�x�����������������������r�r�r�r�����������)�6�B�O�Q�U�O�B�6�)� ����ּμͼϼּ��������������������¿¿¿¿�������������������������˼������'�4�>�@�A�M�O�M�E�@�4�'����������������������ùŹϹȹù�����������FFFFF$F1F3F8F1F$FFFFFFFFFF�������������������������������������������������!�'�%�"�!���������l�S�G�;�3�:�<�B�S�y�����������������~�l����������������������������������������EuEtEoEuE�E�E�E�E�E�E�E�E�E�E�E�EuEuEuEuEEEEEEEEE*E7ECEDEMEGECE7E4E*EE - N + 4 W m 5 C + � � 6 K T , Z W B ^ ; O c ; ) Q % 3 ; P L a L P  V @ E ; . V 9 A , b g S t T =  , <  o O 8 5 F ; f M X 6 H ; b m G F 4 W R 6 Z g < D c +    N  V  �  �  �  p  �  I  
  �  {  "  �  I  �  �  l  n  �  �  �  �  �  �  �  �  R  q  �  +  _  �  �  �  �  �  �  .  +  E  �  �  	  c  �    �    k  O  z  >  �    )  �  �  �  �  '  �  �  �    �  \  �  �    J    �  3  ?  �  �  �  j  U  ?^  ?^  ?^  ?^  ?^  ?^  ?^  ?^  ?^  ?^  ?^  ?^  ?^  ?^  ?^  ?^  ?^  ?^  ?^  ?^  ?^  ?^  ?^  ?^  ?^  ?^  ?^  ?^  ?^  ?^  ?^  ?^  ?^  ?^  ?^  ?^  ?^  ?^  ?^  ?^  ?^  ?^  ?^  ?^  ?^  ?^  ?^  ?^  ?^  ?^  ?^  ?^  ?^  ?^  ?^  ?^  ?^  ?^  ?^  ?^  ?^  ?^  ?^  ?^  ?^  ?^  ?^  ?^  ?^  ?^  ?^  ?^  ?^  ?^  ?^  ?^  ?^  ?^  ?^  8  2  -  (  "        "  *  /  2  4  7  9  ;  =  ?  @  B    
      �  �  �  �  �  �  �  �  �  �  �  �  x  d  O  ;  �  �      *  /  &    �  �  �  �  w  H    �  y    z      �  <  s  �  �  �  ]  ;    �  �  @  
�  	�  	3    �     �  �  �  �  �  �  r  Q  +  �  �  �  �  �  �  �  G  �  �  �   �  �  �  �  �  �  �  �  �  �  �  y  p  f  ]  S  J  @  7  -  #  {  |  v  l  _  M  7    �  �  �  �  [  5  	  �  �  C  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  z  g  P  9  �  `   �  �  �  �  �  �  }  g  N  1    �  �  �  d    �  �  U  ?  "  �  �  �  �  �  �  �  �  �  ^  6  	  �  �  a    �  w  $   �  �  �  �  �  �  �  ~  r  f  Z  O  D  9  .  #  �  �  �  [  &  �  �  �  �  �  �  n  U  8    �  �  �  �  k  L  (  5  E  V  ]  S  I  >  4  )        �  �  �  �  �  �  p  Q  1    �  �  �  �  �  �  �  u  s  t  v  n  ]  M  :  %    �  �  �  �  �  �  �  �  �  �  �  b  @    �  �    B  �  �  b  =  �  X          �  �  �  �  �  �  �  �  �  �  �  {  i  V  D  1  J  H  G  E  B  9  0  &        �  �  �  �  �  v  S  0    �  �  �  �  �    
        �  �  �  r  (  �  C  �  �  3  S  F  8  +        �  �  �  �  �  �  �  r  m  m  m  n  n  B  6  )    �  �  �  �  �  q  T  7    �  �  �  �  o  S  7  �  �  �  �  �  �  �  �  z  c  F  "  �  �  �  �  �  �  �  `  �  �  �  �  �  �  �  �  �  �  �  �  �  y  l  J  !   �   �   �      �  �  �  �  �  �  �  �  �  �  q  c  V  P  C  -    �  a  �  �  �  �  �  �  �  g  B    �  �  �  K    �  5  �  �  e  b  _  \  W  S  O  E  7  *  %  (  ,  -  *  '  $      	  (  .  0  -  $      �  �  �  �  �  �  �  n  V  >  $    �            �  �  �  z  3  �  �  �  �  S    �  `  �  D  �  �  �  �  �  f  2    �  �  �  �  �  �  f  @        /  �  �  �  �  �  �  �  �  {  w  u  o  V  =  %    �  �  �  Z  	      "    
    �  �  �  �  V  #  �  �  z  t  <  �  <  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  n  V  9      �  �  �  �  f  !  �  ]   �  V  �  �  �  �  �  w  `  D  $  �  �  �  w  7  �  q  �  �    ^  k  v  ~  �    w  j  W  <    �  �  �  W    �  �  H    Z  W  S  O  K  G  C  >  5  $      �  �  �  �  n  K  '    U  U  U  U  T  N  I  C  ;  1  &      �  �  �  �  W     �  P  R  Q  M  G  =  .    "    �  �  �  �  j  >    �  S  �  d  `  [  V  Q  L  G  F  E  E  E  E  E  G  M  T  Z  a  g  m  �  �  �                  �  �  �  �  t  H      !  �  �  �  �  �  �  �  �  �  �  �  �  �  q  K  "  �  �  �  q  u  x  x  {  �  �  �  �  |  b  H    �  �  �  �  :  �  �  �  Y  l  w  ~    z  l  W  =  #      �  �  �  ]    �  C   �  �  �  �  �  �  �  x  m  _  P  >  (    �  �  �  �  �  X  &  t  x  |  �  �  �  �    |  x  �  �  �  "  R  h  u  �  �  �  �  �  �  �  �  �  w  h  V  =    �  �  �  F    �  �  R    �  �  �  �  p  V  :      �  �  �  W  #  �  �  @  �  �  ?  )        �  �  �  �  �  �  �  �  �  u  P  3      �  �      �  �  �  �  �  �  �  �  d  7    �  i    �  -  �  W  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  f  D    �  �  �  1  Y  �  �  �  #  E  Y  [  M  3    �  \  �  P  �     �  �  �  �  �  �  �  �  \  )  �  �  q    �  [  �  �    �   t  �  �  �  �  �  �  �  �  �  �  �  �  �  �  ~  n  ]  K  :  (  �  �    !       �  �  �  E  �  �  F    �  �  �  R  �   �  Y  B  ,    �  �  �  �  X  /  �  �  �    \  B  0  F  �  }  �  �  �  �  �  �  �  �  �  �  �  c  F  '    �  �  �  �  \  %  ]  �  �  �  �  �  �  p  E    �  �  n    �    ,  �    �  �  �  �  �  �  |  n  \  G  +    �  �  m  ?      �  �  �  �  �  u  e  S  ;  "    �  �  �  �  x  N    �  �  �  �  �  �  �  �  �  �  U    �  �  E  �  �  %  �  U  �  <  e  &  �  �  �  w  i  Z  I  7  %    �  �  �  �    6  �  �  -   �  C  J  O  ?  0  !      �  �  �  �  �  �          $  5  �  �  �  v  o  h  b  [  U  N  I  F  C  @  >  <  ;  A  G  N  G  M  N  M  G  <  &    �  �  w  >  �  �  j  �  y  �  |  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  i  E    �  �  �  �  U  )  �  �  �  ?    '      �  �  �  �  �  �  q  M    �  �  �  �  d  B     �  o  h  a  Z  R  K  D  =  6  /  $       �   �   �   �   �   �   �  
  
d  
�  
�  
�      
�  
�  
�  
  
R  
  	�  	F  �  �  '  �  A  [  ]  q  }  o  [  ;    �  �  �  R    �  �  R  �  �  !  �      �  �  �  �  �  �  w  X  6    �  �  �  P    �  �  �  1  1  1  1  -  *  %        �  �  �  �  �  �  �  �  �  �             �  �  �  �  �  �    c  C  #  �  �  \  &   �  �  �  �  �  �  �  �  �  �  ~  v  l  V  �  �  :  �  �  g      m  \  K  :  *        �  �  �  �          �  �  �  �  �  �  �  �  �  �  �  �  q  S  5    �  �  �  �  V  %  �  �  p  ]  ?      �  �  �  �  �  y  g  ^  Z  V  H  .  �  �  I    �  �  �  �  �  �  �  �  ~  ]  8    �  �  �  U  $  �  �    �  �  �  �  x  e  Q  7    �  �  �  �  j  J  "  �  �  �  �     �  �  �  �  |  $  �  q    �  V  
�  
x  	�  	t  �    c