CDF       
      obs    8   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type                     	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?`bM���   max       ?�bM���      �  �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       N ��   max       P�;      �  �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       ���   max       =�v�      �  l   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?333333   max       @E������     �   L   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?��Q�     max       @v|�\)     �  )   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @*         max       @N@           p  1�   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @ˣ        max       @�o�          �  2<   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       ����   max       >��      �  3   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A�;   max       B,��      �  3�   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A��   max       B,��      �  4�   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       ?�oR   max       C���      �  5�   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       ?0��   max       C���      �  6�   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max         }      �  7|   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          =      �  8\   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          1      �  9<   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       N ��   max       P&z�      �  :   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�
=p��   max       ?�s�g��      �  :�   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       ���   max       >["�      �  ;�   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?333333   max       @E��
=p�     �  <�   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?��Q�     max       @v|�����     �  E|   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @          max       @N@           p  N<   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @ˣ        max       @�a�          �  N�   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         A5   max         A5      �  O�   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?��a��e�   max       ?�o���     �  Pl            6      5   d  }      B      	   5      �   $         
            �   9                            @      $            9                "      
         x               7      N�5�N�q�N��P:�bO[~�P!�FP���P�;O0�EP��JN�u�N?q;P<W�NňP�I/O�=5N��N ��O#?gO�N\O?�O�}Oè�P&X3NB�?NY�Og:�O&%dN}#qO]�O�NuP9NN�aOٜ�N�H�O��:O M)O��}N�,�O��OT�CO�fP�eN�!N�=jO/�N،�P\�O���O�[O%�O�QPR�N���O�΂���������49X�ě���o%   ;D��;D��;�`B<t�<#�
<u<u<�o<�t�<���<�j<ě�<ě�<���<���<�/<�/<�/<�/<�`B<�`B<�=o=o=o=o=\)=��=��=��='�=0 �=49X=@�=H�9=T��=}�=�o=��=�O�=�\)=���=��-=��
=��T=��=�{=�{=�v�O[gt�����tlg\[OOOOOO����������������������������������������������
'0<?SVIE0
��eegit�����������tjge���
0<Ujqm`UI<0#
�������5[gh_B0)���)5Ng�������t[5zz���������������}z^_fz������������zng^�������������������������� ������������~������������������~��������������������@NZh������������hSE@ )5BNPPH>5)�������������������z{x�������zzzzzzzzzz))5?BNKB5,)()BO[_fcda[OB3*�����


��������)5BCGB?5)������������

��������������������������������������-/<HMKH<3/----------������������������������������������������������������������-+(,4<HP[_[YSaUH<8/-������,)���g^\ht����{thgggggggg��������������������

#,*$#






�������
 
�������� 

�������#/<>HUYZ^`^[T</#��������������������trv~���������������t����
#$'#
 ������~x�����������������~)5:BDHJOMGB5)!�������!�������������"�����"#+0<?IDD<<40#""�~������������������7007;CHTajmnkbaTOH;7yutz�����������zyyyy�����
&/1,#
�����		!)157;<;95/)	�������������������"(+,,)	OLTU_anz|����|zwnaUO)&4��������).)
"#,/0/-#
 -6=@EIH?</#�n�s�s�x�s�n�a�U�U�R�U�X�a�c�n�n�n�n�n�n�'�3�6�>�@�L�U�O�L�@�3�(�'���%�'�'�'�'�0�=�I�M�R�N�Q�I�=�4�0�$��$�(�$�0�0�0�0�S�l�������Ļ��ûл��л����m�_�Q�N�F�S���������������������������������������Ѽ4�@�M�W�f������������r�Y�A�1�%�"�,�2�4���0�I�nŇŇ�ņŀ�n�b�I�����ļĹļ������)�BāĚįİĪĠĒā�h�6�)��������FFF$F.F1F5F=F;F1F.FFE�E�E�E�E�E�E�F���������������������Z�N�B�4�/�4�B�Z�s���/�<�H�T�R�H�A�<�/�&�$�)�/�/�/�/�/�/�/�/�O�[�h�t�t�w�t�h�[�U�O�K�O�O�O�O�O�O�O�O���������(�3�,�*����������ïôïò�����������������������������������������Ҽ@�r�����ü�����f�@������ػл���'�@�������ĿؿտϿǿ��������y�m�f�c�h�w����������������������������5�A�N�P�N�I�A�5�3�0�5�5�5�5�5�5�5�5�5�5�"�/�;�H�T�V�U�T�N�H�;�/�"�������"�������������ؾʾ�����s�n�n�t��������[�c�t�~�t�k�[�B�5�4�9�B�P�[���(�5�F�N�T�R�O�N�A�2�(�������DbDoD�D�D�D�D�D�D�D�D�D�D�D�D{DeDSDTDUDb�������*�8�I�O�Y�d�6������ŶŮŰ��������������������������������������������¤ �Z�[�k�s�{�s�m�g�Z�N�A�5�,�(�!� �(�5�M�ZÓàìù��������ùìàÓÒÇÃÃÇÈÍÓ�Z�\�f�s������s�f�c�Z�M�L�M�O�Z�Z�Z�Z�����������
�����������ùññïù���ҿ.�;�G�T�y�����y�m�T�O�;�.�"�	��	��"�.�!�-�:�?�C�=�:�-�)�!�� �!�!�!�!�!�!�!�!��4�A�J�L�9�1�(����н����������Ľݾ������!�%�+�!�����������������������.�;�G�X�N�S�S�G�;�.�����۾����ʾ��`�m�y����������y�m�`�T�T�U�`�`�`�`�`�`������0�@�A�M�Y�N�A�5�(���������H�U�a�e�d�a�]�U�H�<�/�#���#�(�/�2�<�H�������ɺֺ�������ֺɺ����������������"�)�/�5�6�5�0�/�"�������!�"�"�"�"�����!�#�0�2�+�/�7�/�"����������������0�=�I�R�Q�M�I�@�=�0�$�����������$�0�y���������������������y�s�l�k�e�]�^�l�y����	����+�1�*��������������������𺰺��ɺֺ�������ֺɺ��������������@�M�Y�f�f�r�����r�m�f�Y�Q�M�@�?�<�@�@čĚĦĳĶĿ������ĿĵĳĦĚęĐčĉĉč�����ʼּҼϼ˼ʼ�����������������������E7ELEiE�E�E�E�E�E�E�EiE\ETEPEHE@E7E1E*E7�N�[�g¦²»²©�t�g�[�M�B�8�4�=�N�ûлԻܻ�߻ܻԻлû������������������ûS�x���������x�l�_�S�F�:�-�$�$�-�3�:�F�S���(�)�4�7�@�B�A�4�,�(��� ����	��>�3�'�����˹����Ϲ��3�J�e�n�t�r�Y�>ǔǡǭǭǰǭǭǡǖǔǑǈ�ǃǈǑǔǔǔǔ��*�6�C�O�[�O�C�6�*����������������� q G ? 5 F F F # F ; = H 0 9 = * 3 ? + : W D ! F * N   c < : 6 H E ` $ N N < Y A U F / 9 U ; B R t ) y 2 q 1 X    �    &  �    .  �  �  *  �  j  *  �  y  �  �    d    �  U  �  &  X  H  �  _  �  �  �  �  \  �  ]  �  A  w  f    �  �  y  X    �  u  �  �  �  3  �  a  >  �  ��������
�o=0 �;�`B=D��=���>��<�/=��<�1<�t�=�C�<�>�R=Y�<�h<���=C�=T��=��=,1>"��=�1=+<�=q��=ix�=C�=}�=y�#=��=ě�=#�
=��P=]/=�7L=y�#=���=T��=��=�7L=���=\=���=���=�^5=���>H�9=���=Ƨ�=�j=��>\)=��`=�B	d�B!BBc�B$�pB
�pB&\�B��Bb�B]4B �iB�HB��B��B�B�sB�QB��B
[8B�pB7B��B:�B��BOB?�B�)B��B!�@BD�B�aB��B_B!�jB$��B3�B�B��B΀B��B��B�!B�B,��B�B%��B&A�;B4B�B_�B�BsfB��BT�BobB��B	�wB!B@bB$m�B
^ B&?�B>hBC�B>B �gB�B:+B��B�%B�B��B�>B
G�B��Bj�ByB>GB�}BȵB@B��B�B"=MBF�B�)B��B+�B!�OB$��B��B��B�B��B��B��B>bB;�B,��B�1B%�B@@A��B��B�+B�.B�GB�XB��B��BI�B��A�,�?��=B
��@�~�A�|�@�*`A阨A�`�C���A��BA�O�A���Aѱ�A�T�@���Ar��A���A�e�A���AL��A�¦A�,C���A��`B0�A�_JA���A��mAA6�AѦ�Ad �@wI�A-m+A
4AZ��Aj�AA�I�Aö�@0�qA�֡A��ZB	��A�9A�0�@8q@؆�A�@�M3C���A��$@��_@��A5��?�oRBs^A��<AƂ�?�[+B
B"@��RA�o@@�A�3_AڀFC���A�i�A�`�A�A�~�A��@�KAs>A���A���A��ZAL�"A��DA���C���A��PB?�A�y�A��vA�k�AB�@Aќ�Ac,
@zI�A-A	zAZ�QAj��A���AĀ@-4�A�}A���B
2�A)�A�f�@=��@٪A�~�@��C���A�|*@���@�)�A5�?0��B�A��            7      5   e  }      C      	   6      �   $                     �   :                            A      %            :   	             "               y               8                  3      /   9   7      3         )      =   !            !            +                     #      /      %                  #         )               *               1                           #         '         '      +                           #                                                            )                              1      N�5�NoUN��O��`O[~�O�%�O�O��7O%ԡP)N���N?q;P&z�N���P��O���N�^PN ��O#?gO�pO N&O�}OMՒO�=xNB�?NY�O�QO��N}#qN�:N�T�NuOZ��N�aO��`NE��OP5�O M)Ot��N�,�O�T]OT�CO!NdP�eN�!N��LO/�N،�O3��O���O�[O%�O�QPR�N���O�΂  T  �  �  �    �  	k    ]  �  <  )  �  z    B  �  �     �  �  u  #  ,  �  �    �  �  I  �  �  �  F  �  2  5  1  	�  �  �  �    �  W  �  �  2  4  �  ;  �  7  
y  �  ����`B����<D���ě�<�o=0 �>["�;�o<�/<#�
<#�
<���<�t�=��<ě�<��
<�j<ě�=+<���<���=�O�='�<�/<�/=�P<�<�=0 �=<j=o=��=\)=8Q�=0 �=,1='�=P�`=49X=D��=H�9=y�#=}�=�o=�+=�O�=�\)>   =��-=��
=��T=��=�{=�{=�v�O[gt�����tlg\[OOOOOO���������������������������������������������
#0<=BB=90#����eegit�����������tjge! !!#0<IUZbcb\UI<0#!������)/0+���3.,-15BN[muxwsg[NB=3z����������������~zzkmsz������������zrmk�������������������������� ����������������������������������������������������UUX]ht�����������t[U	)15@KKIB5)��������������������z{x�������zzzzzzzzzz))5?BNKB5,)($$%)6BOX\^`^[UOB63-$������

���������)5BCGB?5)����������

������������ ����������������������������-/<HMKH<3/----------������������������������������������������������������������./1;<BHSSNH<3/......����������g^\ht����{thgggggggg��������������������

#,*$#






�����

������� �

! "%/<EHUY\[WQHB</#!��������������������||~����������������|����
#$'#
 �������|{�����������������)5:BDHJOMGB5)!���������������������������"�����"#+0<?IDD<<40#""�������������������7007;CHTajmnkbaTOH;7yutz�����������zyyyy������
###!
����		!)157;<;95/)	�������������������"(+,,)	OLTU_anz|����|zwnaUO)&4��������).)
"#,/0/-#
 -6=@EIH?</#�n�s�s�x�s�n�a�U�U�R�U�X�a�c�n�n�n�n�n�n�'�3�<�@�K�@�;�3�-�'� ��'�'�'�'�'�'�'�'�0�=�I�M�R�N�Q�I�=�4�0�$��$�(�$�0�0�0�0���������������������������x�m�h�d�l�x�����������������������������������������ѼY�f�r�������������������r�e�U�U�O�T�Y����0�=�I�R�Y�X�U�J�<�0����������������B�O�[�h�t�zăąĂ�t�h�[�O�B�4�/�.�2�6�BFF$F-F1F4F<F4F0F$FFE�E�E�E�E�E�E�FF�����������������������g�Z�I�A�E�T�g�����/�<�H�Q�P�H�@�<�/�'�%�+�/�/�/�/�/�/�/�/�O�[�h�t�t�w�t�h�[�U�O�K�O�O�O�O�O�O�O�O���������%�0�)�(���������ôøóù�����������������������������������������Ҽ4�@�f�������������V�@�4�����	�"�4�����������ƿʿʿĿ¿��������}�n�r�o���������������������������5�A�N�P�N�I�A�5�3�0�5�5�5�5�5�5�5�5�5�5�"�/�;�H�T�V�U�T�N�H�;�/�"�������"�����ʾ׾�޾׾ʾ�����������|�~���������[�a�g�t�|�t�g�g�[�N�B�6�;�B�N�R�[���(�5�F�N�T�R�O�N�A�2�(�������D{D�D�D�D�D�D�D�D�D�D�D�D�D{DxDoDgDiDpD{�����*�5�=�A�<�*��������ŸŻ������������������������������������������������¤ �N�W�Z�d�g�g�g�a�Z�N�A�5�4�(�&�'�(�5�A�NÓàìù����������ùìàÓÇÄÄÇËÓÓ�Z�\�f�s������s�f�c�Z�M�L�M�O�Z�Z�Z�Z������������������������������������"�.�;�G�T�`�c�h�`�Y�T�G�;�.�)�"��!�"�"�!�-�:�?�C�=�:�-�)�!�� �!�!�!�!�!�!�!�!�ݽ����	����������ݽн����ɽнݼ�����!�%�+�!����������������������"�.�:�E�F�9��	�����׾˾þʾھ�	��"�`�m�y���z�y�m�`�\�Y�`�`�`�`�`�`�`�`�`�`������-�=�A�E�A�5�(��������������H�U�a�e�d�a�]�U�H�<�/�#���#�(�/�2�<�H�������ɺѺ������ֺɺ��������������"�)�/�5�6�5�0�/�"�������!�"�"�"�"���	���� �.�1�/�"��	�����������������0�=�I�R�Q�M�I�@�=�0�$�����������$�0�y�����������������������y�r�l�k�h�l�m�y����	����+�1�*��������������������𺰺��ɺֺ�������ֺɺ��������������@�M�Y�b�f�r�j�f�Y�V�M�@�?�=�@�@�@�@�@�@čĚĦĳĶĿ������ĿĵĳĦĚęĐčĉĉč�����ʼּҼϼ˼ʼ�����������������������EuE�E�E�E�E�E�E�E�E�EuEqEiE^E\EZE]EiEnEu�N�[�g¦²»²©�t�g�[�M�B�8�4�=�N�ûлԻܻ�߻ܻԻлû������������������ûS�x���������x�l�_�S�F�:�-�$�$�-�3�:�F�S���(�)�4�7�@�B�A�4�,�(��� ����	��>�3�'�����˹����Ϲ��3�J�e�n�t�r�Y�>ǔǡǭǭǰǭǭǡǖǔǑǈ�ǃǈǑǔǔǔǔ��*�6�C�O�[�O�C�6�*����������������� q E ? 2 F . >  B . = H ' 7 >  * ? + 3 G D  > * N   c  E 6 ( E X - B N 8 Y 8 U   / 9 I ; B  t ) y 2 q 1 X    �    �  �    3  >  }  �  �  j  �  �  �  ,  �    d  �  !  U  �    X  H  %  A  �  �  %  �  �  �  �  U  �  w  �    O  �  U  X    �  u  �  r  �  3  �  a  >  �  �  A5  A5  A5  A5  A5  A5  A5  A5  A5  A5  A5  A5  A5  A5  A5  A5  A5  A5  A5  A5  A5  A5  A5  A5  A5  A5  A5  A5  A5  A5  A5  A5  A5  A5  A5  A5  A5  A5  A5  A5  A5  A5  A5  A5  A5  A5  A5  A5  A5  A5  A5  A5  A5  A5  A5  A5  T  >  &    �  �  �  �  �  �  ]  '  �  �    M    �  �  y  r  w  }  �  �  �    z  s  j  ^  R  C  5  &    �  �  �  �  �  �  �  |  n  t  o  _  O  A  9  6  =  Y  �  �    �  v  �  �  �    %  E  f  �  �  �  �  �  k  3  �  �  '  �  S  �  h    {  w  q  k  c  Z  L  >  /      �  �  �  �  �  |  [  Y  �  �  �  �    T  �  �  �  �  Y    �  �  [    �  g  �  q  �  9  u  �  >  �  	/  	Y  	j  	c  	>  �  �  M  �    �  �  a  �  �  x  9  �  �  r  �  0  0  �      �  s  �  �  "  ^  m  �  Y  \  R  F  8  (    �  �  �  �  X  $  �  �  /  �  �  ]  �  �    Y  �  �  �  �  �  �  �  d  2  �  �  \  �  a  �  �  -  9  ;  ;  :  8  6  3  .  &    �  �  �  �  d  5    �  s  (  )  
  �  �  �  �  d  @    �  �  �  �  �  �  �  ]  4  	  �  l  �  �  u  <  �  �  B  �  �  A    �  �  �  P  �  f  �  W  r  v  x  y  z  y  u  o  g  ]  M  9  #    �  �  �  j  B  1  �  	�  
%  
�  
�  
�      
�  
�  
�  
^  
  	�  	Z  �  w  �  �  ]    3  =  B  A  :  ,      �          �  �  B  �  ?  0  c  }  �  �  �  {  p  a  O  ;  %    �  �  �  �  �  �    A  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  w  n  d  [  Q         	     �  �  �  �  �  �  �  �  |  g  S  3    �  >  �  �  �  �  �  �  �  �  �  �  �  �  }  V  $  �  �  '  �  l  v  �  �  �  �  �  �  �  �  y  Z  :    �  �  �  �  6  �  j  u  s  n  f  [  K  6      �  �  �  �  a  0  �  �  �  6   �  \  �  !  �  �    #      �  R  �    .    �  j  	p  �  �     �  �    '  +       �    /  �  �  '  �  >  �  �  <     �  �  �  �  �  �  �  �  �  t  c  N  :  "    �  �  �  �  d  �  �  �  �  �  {  u  ^  ?       �  �  �  �  y  b  K  4    �  �  �  �  
        �  �  �  �  i  ,  �  }  
  �  
  �  �  �  �  �  �  �  m  >    �  �  N    �  �  j  E  P  p  �  �  �  �  �  �  �  �  �  {  o  d  Z  P  F  <  1  $         O  *      �  �  7  I  <  *    �  �  �  _    �  �    R  h  w  �  �  �  �  �  �  �  �  �  �  �  W  $  �  �  w  D  -  �  �  �  �  �  �  �  x  i  [  M  >  /          �    !  7  s  s  n  h  p  �  �  �  �  �  �  �  y  9  �  �  '  g  o  �  F  >  5  -  $        �  �  �  �  �  �  �  �  �  �  �  �  -  L  c  |  �  �  p  P  (  �  �    9  �  �    �  _      �       )  +  ,  3  <  C  G  F  =  *  
  �  �  %  �  `  �         /  #    �  �  �  �  �  Z  ,  �  �  w  (  �  �  �  1    �  �  �  l  F  !    �  �  �  S    �  �  X    �  �  	q  	�  	�  	�  	�  	�  	�  	}  	I  	  �  }  6  �  �  '  �  �  �  b  �  �  �    t  i  _  T  A  .    �  �  �  �  e  <    �  �  �  �  �  �  �  �  �  �  u  g  V  @    �  �  �  �    �    �  �  �  �  �  k  L  '  �  �  �  r  :    �  �  �  [  %  �  �  �  �  �            �  �  �  �  �  |  #  �  �  \  >  �  �  �  X    �  *  '    !  "  +  $    �  �  S  �  >    W  7    �  �  �  �    h  U  @  )    �  �  �  U    �  A  s  |  �  j  S  7    �  �  �  �  c  =    �  �  �  �  H  
  �  �  �  �  �  c  ?    �  �  �  ^  &  �  �  =  �  I  �  <  2  +  $          �  �  �  �  �  q  O  8  !    *  C  [  O  :  V  �  C  �  �    3    �  �  >  �  Z  �  	�  �  �  �  �  o  M  )  #  -  (    �  �  �  �  i  <    �  �  �  P  �  ;  #  
  �  �  �  �  u  Q  *    �  �  y  ?  �  �  �  T  �  �  �  �  �  �  �  �  r  L  &     �  �  y  J    �  f  X  o  7      �  �  �  �  d  E  #  �  �  �  s  *  �  _  �  �  #  
y  
  	�  	]  	  �  r  d  j  5  �  [  �  o  �  �  )  �  ^  &  �  �  w  `  F  (    �  �  �  L    �  �  E     �  v  9      [  ?  $    �  �  �  �  �  a  7  �  �  G  �  �  ,  �  l