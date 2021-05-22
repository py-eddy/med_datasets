CDF       
      obs    K   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type         ����         	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?h�t�j~�   max       ?�`A�7K�     ,  �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       N �   max       P���     ,  �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       ��l�   max       =C�     ,      effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>��\(��   max       @F���Q�     �  !0   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ���G�|    max       @v�33334     �  ,�   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @(         max       @P@           �  8�   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @��        max       @��         ,  98   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       �.{   max       <�/     ,  :d   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A�\�   max       B4�7     ,  ;�   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A�x�   max       B4�6     ,  <�   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       >4ߗ   max       C��     ,  =�   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       >?�R   max       C��`     ,  ?   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          �     ,  @@   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          A     ,  Al   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          A     ,  B�   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       N �   max       PST�     ,  C�   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�a|�Q   max       ?��&��IR     ,  D�   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       ��l�   max       =C�     ,  F   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>��\(��   max       @F���Q�     �  GH   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ��(�`    max       @v�33334     �  S    speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @)         max       @P@           �  ^�   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @��        max       @�=�         ,  _P   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         B�   max         B�     ,  `|   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?~Ov_ح�   max       ?ҸQ��     p  a�                  7         	            <   �      &               ,                                 #   L            B                  B      
                  
   *   
                        "            ,                              NZ�VNh��N��JOu�N%�O��NaʝN�([O �ON���NTn�O�x�P7GP���NM�O���N��-O&ӎN���N��=P/�O}�N$��O	x<NWD�O��N��=N�NJ��Oo:�N�{"P*#P��O^�&P:�O%5�P0��Nc�PP�O�k�Os�O���PST�N��bO�XNތO%��N�1Oh��N:��O.�qO��MN���N��'O�`N�AoN�+O|�Nv��O_)NP�OU!�O(N���O��kN�T�N���Ne8�N�!O�U�Om?Ny.�N��VN �OF�4=C�<�1<�o<�o<#�
<#�
<o;�`B:�o%   �D���D���D���D����o��o���
��`B�o�o�#�
�#�
�49X�49X�D���D���T���T���e`B�e`B�e`B��t���1��9X��9X�ě��ě���/��`B��h��h��h��h�����+�\)�\)�t��t���w��w�#�
�'',1�,1�0 Ž0 Ž0 Ž49X�49X�49X�49X�49X�49X�]/����������1��E���E��ȴ9�ȴ9��l������������������������������������������������������������� ������������������������������������������������������������������059BLNS[[[VNGB510000���������������������������������MNOU[`gmjg[NMMMMMMMM������������������:?_nz���������znaH::����)49:6������{��! �������������������������������������
�����������
#/36975/# ���#/3<><://##���������������������������
#*,#
������������

����������������������������9<ISUbnu����{nbUI<79COY[ce[OKDCCCCCCCCCC��������������������RTampnmjaZTSLNRRRRRR������������������������	
	������������$)02&�������MNY[gotuvtrlig[RNFMM���#/2@B@<2����z��������������znrsz���
$)+)#
������z�����������������|zW[ht�����������yh[WW��#IUbhkkbU<#�����DO[htmha[YOKDDDDDDDD%/;HT]amuqgneaTH(# % -1-6;6�������9=HakmzzwtmeaTHF=:9GUht���������th[TIGG����5D6��������������������������������������������������#0<INU]XUI<0-#[amz�����������zmaX[�������������������������������������������������������h\OC:667<?COZ\dhlnjh56BO[t������{[OB?95���


�������������������������������o{�������������}hcho������������������������������������������������������������""	')+5BO[hjqskd[SOB6)'����*21)���������)57�����)+/0/-)=BNR[gtz~tg[[ZNB====sy���������������|rs��������������������Z[fhmtv~toh[USZZZZZZ����������������������������������������gt������������tmgcdgstz����������~tpppqsot������ttmmoooooooo

#*&(&#






//:<>A><1/-+////////muz������������zwnlm��ۼ�������������������㾱�����������������ʾ˾ʾʾ�����������������������������������������������t�q�g�[�O�H�H�N�[�^�g�j�u�t���������Ŀѿݿ�ݿѿĿ������������������M�&����'�.�A�M�Y�f�m�����������f�Y�M�#�!��#�/�<�=�H�S�H�<�/�#�#�#�#�#�#�#�#����y�p�y�|�����������������������������!����!�-�:�F�S�_�`�e�_�S�F�F�:�-�!�!ĿĳĳĦģĢĦħĳĵĿ������ĿĿĿĿĿĿ�s�k�f�Z�O�Z�a�f�s�t�~�w�s�s�s�s�s�s�s�s�M�3�3�,�*�0�,�'�(�4�M�h�������s�g�Z�M�ù��������������ܺ�'�@�F�C�:�7�&���Ϲü�������'�@��������������n�Y�@��������������������������������������������������������������	�� �'�-�"��	����������������������������������������������������ŹŶŷŹ�����������������������ҽ��ݽҽսݽ����������������н̽Ľ����Ľͽнݽ޽�����ݽннн��	� ���������������������/�P�Q�K�;�%��	�g�a�`�Z�\�`�g�s���������������������s�gF=F;F=F?FJFVFcFdFcFVFUFJF=F=F=F=F=F=F=F=�������������������Ǽʼ˼ͼмӼڼ׼мʼ��Y�N�Y�Z�e�r�{�v�r�e�Y�Y�Y�Y�Y�Y�Y�Y�Y�Y������߽ݽ۽ݽ޽���������'�����������������������������������������̺L�I�A�@�:�@�C�L�Y�_�e�r�t�z�r�e�`�Y�L�L�	���������	�
�� �"�%�"��	�	�	�	�	�	�5�(�.�5�A�N�U�Z�g�s�v�y������s�g�Z�N�5�g�f�Z�X�Q�N�L�N�Z�b�g�s���������|�s�g�g�4���������(�A�f�s�����������f�M�4���������������-�F�_�v�{�y�[�U�:�!�ֺ����������}������������ľþʾϾʾ����������ѿ����������������ѿٿ����8�>�(����y�r�r�t�u�w�y�������������������������y���y�c�]�l�t�������ݽ����ϽȽʽսнĽ������������������������������������������������o�i�r���������������������������-�5�A�N�g�~�������������s�g�Z�N�5�-�(�-���	�������(�5�D�N�V�X�Z�N�5�(��
�������������
����*�7�>�<�6�/�#��
�]�Y�e�������!�.�$�����ּ������r�]������������������������������������������������������������������������߻!������!�'�-�:�=�B�@�:�:�3�-�#�!�!�������������������	�
�������
����ÓÑÈÉÓÓÓØÞàãàÓÓÓÓÓÓÓÓƚƎƃ�}ƀƎƚƧƳ����������������ƳƧƚ����������$�)�*�$�������������������z�����{�m�d�`�T�G�;�0�2�;�<�G�L�T�`�m�z�l�_�R�R�Y�h�l�v�x���������������������l��� ������)�4�,�)����������ݿҿѿͿѿٿݿ���������������ݿݻû��������ûлܻ����$��������ܻ����	�����	���"�.�/�:�;�A�;�/�"���������!�"�!� ������������n�k�h�j�n�t�{�}ŇŏŔŖŗŝŠŔŇ�{�n�nŇŇŇŇŐŔŠŭŵűŭŠŕŔŇŇŇŇŇŇ�m�g�a�T�I�A�?�G�H�T�a�m�z�����������z�m������ùààíñìù��������	�����"��
��������	����"�.�9�=�>�;�.�"�n�d�a�U�P�H�H�Q�U�a�n�zÇÑÎÇÆ�{�z�n�	�������������	�������
�	�	�	�	ǔǈ�~�p�e�b�Y�W�o�{ǈǔǣǪǰǲǯǭǡǔ�������������������������������������������������������ù˹ϹӹԹϹù�����������ĳįĳĵĿ����������������Ŀĳĳĳĳĳĳ����������������������������������������Ěčć�u�r�{āčĚĦĹ������������ĳĦĚ¿½²«¨°²¿����������������������¿�����#�/�<�>�<�;�/�#��������E*E*E&E$E*E7ECEPETEVEPEHECE7E*E*E*E*E*E*EED�D�D�EEEEE!EEEEEEEEEE�������������������������ʾ׾�۾׾ʾ��� 2 / - P w # h = 9 ^ ; N N C ;  V N @ ( O J ^ t K ? = / K 8 T D J , T h E d T 6 2 E l S R p n � @ q  > S g * _ + F Y 9 E V ) � ; L ; R e G i L 1 k 5    l  q  �  y  �  �    �  "  �  j  Z  �  T  r  3  �  �    �  l    n  �  v  M  �  �  c  �  �  �  }  �  �  �  )  �  �  ~  �    �  �  9  ]  �  �  �  w  k  �  �  �  v    '  d  �  �  �  �  I      �  �  �  8  �  Z  �  �  /  �<�/<�C�<#�
;��
<o�49X;D��:�o�o���
��`B���
�}�.{��`B�'t��t����㼼j�]/������t���1��j���㼣�
��`B��C���/��C��T���Ƨ�#�
�H�9��h��Q�o�D���T���,1�]/����8Q�#�
�8Q�P�`��w�]/�,1�D�����
�L�ͽP�`�e`B�y�#�<j�]/�D����C����-��%�}�H�9��-�D����\)��1��-��xս��ͽ\���ٽ����\)B�B4�7BҒB/�B+�B ��BZ�B�B,�B�^B��BT�B��Bk�B��B)�Bc�B�RB�>B)�IB�AB�PB��B'��B��B ��A�P/B��A�\�B�yB	QBQeB��B$6�B*@NB��B%�B(MA�
jBn�A���B[�B-��B�7B��B&�A�;AB]�BƾB�qB1PfBYBB��B�?B)�B ��B!��B�B�pB�lB9�B�wB$eB�GB/�B��B�uB��Bp�B
�gB
|�B
�BY�B��B�B�=B4�6B��B��B;B �IBIB�$B,�IB�B��B@�B:B?�B@�BD�Bo�B�>B��B)��B�3B�zB4�B'D�B$�B f�A��|B�PA�x�B�/B�7B.=B@NB$��B*=�B��B&DQB�KA��0B@�A��4B��B-?mB��B��B&�PA��BZ�B��B�BB1t(BKLB��B��B*M�B ��B!�B�IB�BFB�B�7B9VB	�BBK�B��B��BZIB
F�B
?B	��B@�B�nB�SA&AM�yA�8�A��Ax�	@׍�A�!Ao�<@}A��pAAsMA=�>���@�w�AI@A�SRA���A���A.�KA*r�A�A�;^C��@���?�+0A0IBؙ?�h�A��A�l�A��nA=��@`�AJ��A�Ap��A#i�A��dA�N~A��HA��LA�i�@�\�AЦ�A��@p�mA��dA�ڌB�B	�AhC @�P�A�X�A#U@�3�A�E�@a�1A�.A��A��AЗHA^�A�-�AY�[B�A�`=>4ߗA�Z�A���A�+�A�ˉA��aC���C�b�AM��A�AL�_A���A�~�Ay�9@ِ�A��Ap��@{��A�}�AA-+A>��>���@�ݛAH�yA�z�A��|A��VA/D9A+SA���A� C��`@�j"?�'�A148B�t?�^A��)A��A�WAB�@d?AIBAf�An�#A �RA���A�HA�vuA���A��A�AКA��@s�9A�A��B��B	�Ai��@��@AՂ(A}(8@�eA��#@c�A�}�A�,�A��A�Q�A]j�A��BA[<�B>A�}#>?�RA�y�A�\�A�}�A�`YA��C���C�YMAM�                  7         
            <   �      &               -                                 #   M            C                  B      
                  
   *                           "            ,            	                                    %                  '   1   9                     /                                 +   7      1      1      -            A                           !                           -                                                                              !   !   %                     %                                 )   %            1                  A                                                                                                NZ�VNh��N��JO�aN%�ORf�NaʝN�([O �ON���NTn�O�v)O�A�P��NM�O|�:NK(�O�	N���N��qO�]�O}�N$��N�C�NWD�O��N��=N�9+NJ��Oo:�N�{"P�7P�JO^�&Oo�tO%5�P0��Nc�POf��O�k�OQ�JO��PST�N��bO�XNތN�NN�1Oh��N:��O.�qO���NM�mN��'O�`N�AoN�+O|�Nv��O_)NO�R3O%RN��N���O@�?Np�NE4SNe8�N�!O�U�Om?Ny.�N��VN �OF�4  �  �  �  �  �  )  R  l    +  l  �  _  s  �  �  /  �  �  x    F  �    �  <  �  �    �  g  �  �  4  v  �  �  �  #  �    �    ;  �  Y  �  �    �  �    �  S  4  �  �    ]    �  m    �  k  D    x    N  +  D  �  K  3=C�<�1<�o<u<#�
�t�<o;�`B:�o%   �D����o���
��7L��o�o�ě��o�o�#�
���
�#�
�49X�D���D���D���T����o�e`B�e`B�e`B���
�P�`��9X�o�ě��ě���/��P��h������h�����+�#�
�\)�t��t���w�',1�'',1�,1�0 Ž0 Ž0 ŽT���<j�H�9�49X�P�`�8Q�ixս���������1��E���E��ȴ9�ȴ9��l�������������������������������������������������������������������������������������������������������������������������������059BLNS[[[VNGB510000���������������������������������MNOU[`gmjg[NMMMMMMMM������������������AHanz����~znia\UHC?A���������������! ������������������������������������������������
#/26874/#
��#/3<><://##�������������������������
"
�������������

����������������������������;<IVbns{�{nbYULI<9;COY[ce[OKDCCCCCCCCCC��������������������RTampnmjaZTSLNRRRRRR������������������������	
	������������$)02&�������MNY[gotuvtrlig[RNFMM���&-/>@;0�������������������z||����
$)+)#
��������������������������W[ht�����������yh[WW��#IUbhkkbU<#�����DO[htmha[YOKDDDDDDDD+/;HQTVZ\]]KH;6.+))+ -1-6;6�������:;@HagmvusnmcaTHC?;:LZht���������th[VLJL����5D6��������������������������������������������������#0<INU]XUI<0-#^ahmz����zoma^^^^^^�������������������������������������������������������h\OC:667<?COZ\dhlnjh68BO[tz����zf[OHB@:6��

���������������������������������o{�������������}hcho������������������������������������������������������������""	')+5BO[hjqskd[SOB6)'���)+,)������������
������&)+,))
=BNR[gtz~tg[[ZNB====|����������������vv|��������������������U[\hrtztkh[XUUUUUUUU����������������������������������������gt������������tmgcdgstz����������~tpppqsot������ttmmoooooooo

#*&(&#






//:<>A><1/-+////////muz������������zwnlm��ۼ�������������������㾱�����������������ʾ˾ʾʾ�����������������������������������������������t�g�[�R�N�J�K�N�[�d�g�h�t�t���������Ŀѿݿ�ݿѿĿ������������������;�4�(�'�%�'�0�4�@�M�Y�f�r�{�|�r�q�Y�M�;�#�!��#�/�<�=�H�S�H�<�/�#�#�#�#�#�#�#�#����y�p�y�|�����������������������������!����!�-�:�F�S�_�`�e�_�S�F�F�:�-�!�!ĿĳĳĦģĢĦħĳĵĿ������ĿĿĿĿĿĿ�s�k�f�Z�O�Z�a�f�s�t�~�w�s�s�s�s�s�s�s�s�8�5�.�+�1�-�)�4�A�M�f�������s�e�Z�M�8�ù����������ù�� ���������ܹϹüM�@�4�'�!���!�'�4�@�Y�r���������s�f�M�������������������������������������������������������������	���!�!���	��������������������������������������������������źŹŶŷŹ�����������������������ҽ��ݽҽսݽ����������������Ľý½Ľнݽ�����ݽнĽĽĽĽĽĽĽ��	������������������	�&�7�D�G�@�4�"��	�g�a�`�Z�\�`�g�s���������������������s�gF=F;F=F?FJFVFcFdFcFVFUFJF=F=F=F=F=F=F=F=�������������������ʼ˼ϼѼּؼּּϼʼ��Y�N�Y�Z�e�r�{�v�r�e�Y�Y�Y�Y�Y�Y�Y�Y�Y�Y������߽ݽ۽ݽ޽���������'�����������������������������������������̺Y�R�L�E�@�=�@�I�L�W�Y�e�m�m�e�[�Y�Y�Y�Y�	���������	�
�� �"�%�"��	�	�	�	�	�	�5�(�.�5�A�N�U�Z�g�s�v�y������s�g�Z�N�5�g�f�Z�X�Q�N�L�N�Z�b�g�s���������|�s�g�g�4���������(�A�f�s������������f�M�4�кź��ºɺ����-�F�S�[�\�W�:�-�����о��������}������������ľþʾϾʾ��������ݿֿǿĿ������Ŀѿ����
��������ݿy�r�r�t�u�w�y�������������������������y���y�c�]�l�t�������ݽ����ϽȽʽսнĽ������������������������������������������������������������������������������������-�5�A�N�g�~�������������s�g�Z�N�5�-�(�-����
�����(�5�A�B�N�S�U�O�A�5�(��
�������������
����)�6�=�:�4�/�#��
�]�Y�e�������!�.�$�����ּ������r�]������������������������������������������������������������������������߻!������!�'�-�:�=�B�@�:�:�3�-�#�!�!�����������������
���
�	�������������ÓÑÈÉÓÓÓØÞàãàÓÓÓÓÓÓÓÓƚƎƃ�}ƀƎƚƧƳ����������������ƳƧƚ����������$�)�*�$�������������������z�����{�m�d�`�T�G�;�0�2�;�<�G�L�T�`�m�z�k�_�S�S�Z�j�l�x�����������������������k���	���)�1�*�)������������ݿҿѿͿѿٿݿ���������������ݿݻû��������ûлܻ����$��������ܻ����	�����	���"�.�/�:�;�A�;�/�"���������!�"�!� ������������n�k�h�j�n�t�{�}ŇŏŔŖŗŝŠŔŇ�{�n�nŇŇŇŇŐŔŠŭŵűŭŠŕŔŇŇŇŇŇŇ�m�g�a�T�I�A�?�G�H�T�a�m�z�����������z�m����òíøþý�������	������������ҿ��	����������	��"�+�.�6�;�<�;�.�"��U�T�M�U�[�a�n�z�|Ä��z�n�a�U�U�U�U�U�U�	�������������	�������
�	�	�	�	ǈǂ�{�u�o�l�g�o�vǈǔǜǡǦǬǯǫǡǔǈ�����������������������������������������ù����������ùȹϹѹҹϹùùùùùùù�ĳįĳĵĿ����������������Ŀĳĳĳĳĳĳ����������������������������������������Ěčć�u�r�{āčĚĦĹ������������ĳĦĚ¿½²«¨°²¿����������������������¿�����#�/�<�>�<�;�/�#��������E*E*E&E$E*E7ECEPETEVEPEHECE7E*E*E*E*E*E*EED�D�D�EEEEE!EEEEEEEEEE�������������������������ʾ׾�۾׾ʾ��� 2 / - P w  h = 9 ^ ; J 3 " ;  k ( @   J J ^ o K ? = ) K 8 T C R , @ h E d < 6 $ ? l S R p c � @ q  5 G g * _ + F Y 9 ' 0  � 6 ? 7 R e G i L 1 k 5�y  l  q  �  ]  �  �    �  "  �  j  +  �  f  r  �  �  S    �  k    n  `  v  M  �  �  c  �  �  �  �  �  �  �  )  �  �  ~  �  8  �  �  9  ]  �  �  �  w  k  e  d  �  v    '  d  �  �  �  b  �    �  �  b  �  8  �  Z  �  �  /  �  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  �  �  w  _  F  .    �  �  �  �  �  �  �  x  ]  B    �  �  �  �  �  {  w  r  o  k  h  d  `  Z  S  M  G  6  "     �   �  �  �  �  �  �  �  �  �  �  z  r  m  h  d  b  a  c  u  �  �  �  �  �  }  n  [  K  ;  -       
     �  �  �  �  �  �  5  �  �  �  �  �    {  w  r  n  k  i  g  f  d  b  `  ^  \  [  �  4  z  �  �      )  !    �  �  �  H  �  �    �  �  `  R  K  C  <  4  *  !        �  �  �  �  �  �  �  �  �  �  l  c  Z  R  E  9  ,      �  �  �  �  �  �  �  �  �  �  	        �  �  �  �  �  �  �  �  �  �  �  |  j  P  5     �  +  "      	        �  �  �  �  �  �  �  {  b  H  /    l  n  p  r  t  t  n  h  c  ]  V  N  F  >  6  .  %        �  �  �  �  �  �  �  �  j  L  ,    �  �  �  �  z  P  %   �  �  �  �  0  Q  ^  ^  T  5  
  �  �  J  �  �    �  �  �  "  �  	  	�  	�  
�  
�  ?  m  j  Y  >    
�  
z  
  	x  �  s  �  �  �  �  �  �  �  �  �  �  �  �  |  u  m  ^  @  "     �   �   �  �  �  �  �  �  �  ]  6    �  �  �  l  J  .  /    �  .  �  (  )  +  ,  .  /  1  3  5  6  :  >  C  G  L  P  S  W  [  ^  �  �  �  �  �  �  v  Y  ;    �  �  �  @  �  d  �    8   �  �  �  �  �  �  �  �  j  R  :     	  �  �  �  �  {  7  �  �  [  p  w  q  b  S  F  9  ,        �  �  �  �  L    �  �  �  �  �           �  �  �  �  {  L    �  \  �  j    �  F  2  #    �  �  �  �  �  �  q  X  ?  #    �  �  f  '  �  �  �  �  �  �  �  �  �  �  �  l  U  6    �  �  �  l  =                  �  �  �  �  �  l  I  H  V  i    N    �  |  c  K  2    �  �  �  �  �  g  A  %    �  �  �  �  �  <  6  0  *           �  �  �  �  �  �  �  �  �  |  z  w  �  �  �  {  n  b  T  F  9  (      �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  i  @    �  �  �  ^    �  �  @           �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  n  U  9    �  �  �  h  /  �  �  �  q  j  ~  g  b  ]  X  S  N  I  G  G  G  G  G  G  D  :  1  '      
  �  �  �  �  �  ~  u  c  G     �  �  g  <  K  ,  �  �  �   �  ,  �  �  �    p  �  �  �  �  �  )  �  *  �  -  s    �  %  4  1          �  �  �  �  l  J  9  (    �  �  3   �   J  7  0  /  >  H  Y  m  v  s  f  M  /    �  �  ~  -  �  |  /  �  �  �  �  �  �  �  z  m  _  P  A  -     �   �   �   �   �   �  �    ~  r  `  N  3  	  �  �  6  �  �  `  #  �  ^  �  �  �  �  �  �  �  �  �  �    u  p  k  f  k  v  �  �  |  e  O  8  �  �  �    	        "         �  �  �  �  X    �  �  �  �  �  �  s  J    �  �  d  *  �  �  �  �  �  h    �  +  �  �    �  �  �  �  �  �    i  U  B  /    �  �  �  �  P  �  �  �  v  ^  H  /    �  �  �  u  
  �  1  �  [  �  {        �  �  �  �  �  �  �  }  L    �  Q  �  Y  �  I  �  h  ;    �  �  �  q  H    �  �  �  W  !  �  �  A  �  t  �  e  �  �  �  �  �  �  m  Q  6  !      �  �  �  �  �  _  7    Y  Q  G  :  +      �  �  �  �  �  p  6  �  �  x  1  �  �  �  u  k  c  [  _  �    o  W  4    �  �  ;  �  �  I  �  �  �  �  �  �  �  �  �  �  ~  o  Z  A  '    �  �  �  �  �  �        �  �  �  �  �  �  i  C    �  �  �  �  5  �  �  m  �  �  �  �  �  �  �  �  y  s  p  q  r  p  g  ]  T  G  :  ,  �    {  k  [  I  7  "    �  �  �  �  �  �  v  [  ?     �  f  ~  p  d  X  F  1      �  �  �  p  (  �  l    �      �  �  �  �  �  �  �  �  �  x  k  Z  I  7  *    &  1  A  R  S  F  9  "    �  �  �  �  y  U  1    �  �  ~  K    �  _  4  .  "    �  �  �  �  �  �  �  r  U  6    �  �  n  *   �  �  �  �  �  �    z  n  V  N  8    �  �  r  2  �  �  w  H  �  �  �  �  �  |  }  ~      �  �  �  �  �  �  �  �  �  �    �  �  �  �  �  �  �  �  �  n  U  8  &    �  �  g  %  �  ]  T  J  A  7  *        �  �  �  �  �  �  �  �  �    #            �  �  �  �  �  w  R  !  �  �  /  �  �  K  �  \  v  x  �  �  �  �  }  ^  (  �  �  /  �  b  <      �  �      m  g  \  R  F  8  '       �  �  �  X    �  Y  �  z  �  �              �  �  �  �  f  0  �  �  o    �    �  �  �  �  �  �  �  �  �  n  [  G  3      �  �  �  �  �  �  <  f  j  i  ^  I  0    �  �  �  @  �  j  �  X  �    i  A  B  B  C  C  B  <  6  0  *  $            �  �  �  �  �  �    
         �  �  �  H     �  d    �  9  �  .   �  x  p  g  ^  T  F  7  (    �  �  �  �  �  �  p  ]  I  5  !    �  �  �  �  z  u  o  �  �  �  �  �  j  A    �  �  >  �  N  I  D  <  ,    �  �  �  �  l  B    �  �  @  �  T  �  $  +    �  �  �  �  �  �  �  �  {  b  G  %    �  �  a  Q  U  D  3  "    �  �  �  �  �  �  �  b  D  %    �  �  �  �  b  �  �  �  �  �  �  �  ^    �  A  �  2  �  �  O  �    Y  �  K  C  :  2  &        �  �  �  �  �  �  Z    �  �  w  K  3    �  �    M    �  �  X    �  �  J  �  }  �    )  �