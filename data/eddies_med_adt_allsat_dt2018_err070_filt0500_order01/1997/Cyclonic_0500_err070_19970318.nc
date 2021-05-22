CDF       
      obs    S   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type         ����         	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?`bM���   max       ?�?|�hs     L  �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       M��`   max       P��?     L  �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       �Ƨ�   max       =]/     L   D   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>�G�z�   max       @F%�Q�     �  !�   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ���G�|    max       @v|          �  .�   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @+         max       @R            �  ;�   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @�_        max       @�c�         L  <(   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       ��   max       =,1     L  =t   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A��5   max       B/��     L  >�   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A��   max       B/�     L  @   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       >���   max       C���     L  AX   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       >��G   max       C���     L  B�   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          T     L  C�   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          C     L  E<   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          ?     L  F�   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       M��`   max       P�Q�     L  G�   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?��`A�7L   max       ?�s�PH     L  I    speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       �Ƨ�   max       =]/     L  Jl   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>�G�z�   max       @Fp��
>     �  K�   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ���G�|    max       @v|          �  X�   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @)         max       @R            �  e�   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @�_        max       @��`         L  fP   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         C[   max         C[     L  g�   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�t�j~��   max       ?�j��f�B     �  h�                     	               '            F   )   	   !            (      -   	      
            
               
      	   #         
         ;         %      #   %         2   !               %         
         S                  J         "                     N�,O}�;OyN N�5?M��N��2N�U&O�sO11�NBp�O��RO1�%Oe!vO/�{P��?PQ�UOPP��N�&�N5��O Y}O��jN��&PZ�N��N$��O6vsO�"�O�5�O���O��N��^N+��O�_�O�˿N���O:��NVN�P}YO��NB�Nr��OZůOPFPH-N�}�N���O��O�GO�]
O��Nxp�N�.�P;,hO��#O?�N#HQN0L�OD]�O���M��`N�{�NGrAN.IO7ݨP9�O���O�2�O��N�3�O�ҬO*�=N� �OK��O��kNB�N>I N6zfOV�O,�N�"�O�9=]/<���<�o<49X;�`B;D��;D��;D��%   %   ��o��o�D�����
�ě��ě��o�t��t��#�
�#�
�49X�49X�49X�D���D���T���T���e`B�u��o��o��C���t����㼛�㼣�
���
��1�ě��ě����ͼ�`B��h���o�+�+�+�+�C��C��t���P��P������w�#�
�0 Ž49X�49X�8Q�8Q�@��@��@��@��D���T���Y��Y��ixսm�h�q���q����o��o��o��������Ƨ�Ƨ�������������������������������������������������������������������������������������������������������������������������$),))")("��(-)&##�����>BGN[gjtxxtkg[NB??6>#/09/#)6BKV[^gf[B6)')+5BNRNOMNKB5+)&'%'
#/48;93/#
����������������������#0U{������{bI0�����
#/689C/(��������������������������������������������������	!"	��������(/;C?;8/'#((((((((((lpz������������znjfl�������2+����� )6BOZ[\[WOB66*)    GOa�����������maPHDG6BOOWOONB96666666666GHNUYXVUHGECGGGGGGGG,69BEO[htux{th[OB6/,��������������������BJO[hqsushdb[B9<7<<Bz�������������zuqouz��������������������tz�������zvuvstttttt������������������������������������������	*6CKMB6���������������=BDNTX^gmsug[NJHF=<=���������������������������������!)5:;;6)��lnsz}~~znmjfllllllll$��&'#/@HKHBB:/#X[h����~{xtrhc[YSRQX����������������������������������������kmz�������zummkkkkkk������������������������������������������������������������������������#/16/'#����������������������������*.������)B<HZ_XXSN5)"!#0/4&)|���������������zz||�����������������������

�������������{������{naWUNOU^adn{������������������������������������������������������������15BBBA?>53)+11111111rz�����zwqrrrrrrrrrrXanz���������{~na_UX��������
��������
#0FINWUI;10#

t�����������tqmlkklt��������������������./5<HPSSOH<7/.......Uanz��������zsmaURSU�����
!!
�����:BO[ghjhhc^[OKGB>6::MR[gt������tg[SNJLMMNU[gt{�������tg[PMJN
#%#
 #09840#           )0<>CIII<10)))))))))dgjt�����������wmfd������������������������������������������	 �������E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E��#�������������'�/�5�<�>�F�J�K�F�<�#�нȽĽ����������Ľнݽ�����������ݽ��U�R�O�P�U�a�c�b�a�]�U�U�U�U�U�U�U�U�U�U�M�I�A�:�4�(�$�(�,�4�A�M�Z�]�^�Z�W�P�M�M�3�-�'������'�3�4�9�3�3�3�3�3�3�3�3�3�2�0�3�9�@�G�L�X�Y�c�d�Y�Q�L�@�3�3�3�3��ںֺκֺ������������������u�t�u�v�s�s�t�����������������������f�a�Z�N�N�X�Z�c�f�s�x�������������s�f�t�r�p�s�t�t�t�t�t�t�t�t�t�t�t�F�:�!��%�)�:�F�S�_�x�����������y�l�_�F��������������������*�6�3�*�$�����������ſſ�����������������������������������	��"�/�;�H�K�L�;�5�"��	�����������X�O�<�!��(�Z�������������������N�G�O�Z�j������������������������~�Z�N����������������"�*�+�6�B�6�*������㾾�����f�M�A�7�6�H�f���������������"� ���	��	��"�/�0�1�;�=�;�/�"�"�"�"�����"�)�/�7�/�"�����������������������������������������������ùà×Ç�n�a�S�Q�U�b�n�zÇÓàù������ù�f�a�\�[�Z�f�n�r�t��������r�p�f�f�f�f�N�5�������(�A�n�s�����������������s�N�a�[�a�a�n�z�}�z�u�n�a�a�a�a�a�a�a�a�a�a����������������������������������������������������������������������������"�������/�;�T�a�i�c�Y�T�Q�H�>�;�"��	�� ��	��.�;�Y�`�m�y�������`�G�;���ý�����������6�C�B�D�2�)�%�������ƋƁ�h�d�f�h�sƁƎƚơƳ����ƿƿƳƧƚƋ�g�\�`�g�s�����������������s�g�g�g�g�g�g�M�D�@�4�3�4�8�@�B�M�T�Y�Z�Y�M�M�M�M�M�M���y�`�J�D�G�T�T�`�y�������ƿʿʿĿ������G�@�"��	���޾���	�"�2�G�`�r�u�m�T�G�[�S�O�J�O�Y�[�h�t��{�t�n�h�[�[�[�[�[�[�g�a�g�l�s���������������������������s�g�@�9�>�@�G�M�Y�[�`�Y�Y�M�@�@�@�@�@�@�@�@���~�e�Y�M�U�T�]�w���������ͺغںֺ������(�4�9�A�M�T�f�s���������������f�Z�A�(���������������������������������������˿������������ĿſʿɿĿ������������������H�<�#���
���"�/�<�H�U�a�o�r�n�a�U�H�нϽͽƽнݽ��������������ݽп�	���ھپ����G�m�q�������y�`�G�.��U�a�n�r�z�z�|�z�n�n�a�U�H�E�@�D�H�U�U�U���������������������������������������������������$�+�0�4�4�0�$��������������������������������������������T�;�.�#���'�-�.�9�G�T�m�}�����y�m�`�T�x�v�l�`�Z�Y�_�l�x���������������������x�6�5�,�5�6�B�K�O�Q�U�O�B�6�6�6�6�6�6�6�6��������'�4�@�M�U�M�D�@�4�,�'�������r�f�d������ʽ�!�2�5�2�!�����ʼ��u�vÇÓìù�������������ìàÓÇ�u�I�=�0�*�+�0�=�I�V�b�o�{�ǈǊǈ�{�o�V�I�ɺź������ɺֺ����ֺɺɺɺɺɺɺɺ��0�&�$�-�0�5�=�D�B�=�0�0�0�0�0�0�0�0�0�0���������Ŀѿѿݿ��ݿڿѿĿ������������t�Y�I�C�F�[�tāčĦĳľ��ĿĳĢģĚĉ�t�@�:�?�@�L�L�U�T�N�L�@�@�@�@�@�@�@�@�@�@�U�P�I�C�I�U�^�b�n�u�{ŇŇŃŀ�{�t�n�b�U����������������������������������������ĿĺĿ��������������ĿĿĿĿĿĿĿĿĿĿ�H�A�;�5�2�:�;�C�H�T�Y�_�_�a�m�t�g�a�T�H�&��������'�@�Y�m�k�m�r�������Y�&�лû��������������ûлֻ���������ػ�ĸĻ�����������
��#�,�*���
��������ĸ������������������
��"�#�(�#���
����FFFFFFFF$F1F5F7F1F(F$FFFFFF�������������ùϹܹ����������ʹ�����E*E'EEEEEEE"E*E7ECEPETEZE]EPEDE7E*����������������������������������������������������������"�'�&�)�,�)�����
� ����
��%�/�<�H�J�O�R�V�U�H�<�/��ݽѽ۽ݽ������������ݽݽݽݽݽݽݽݽ����������ĽɽνĽ������������������������� ������&������������{�s�{łłŇņřŭŹ����������ŹŭŔŇ�{�������x�x�������ûлܻ��߻ܻлû������!��!�"�.�.�:�G�J�M�G�@�:�.�!�!�!�!�!�!�S�H�L�S�[�`�l�y�������������|�y�l�`�S�S 4 . % � : { 9 @ @ 2 o = I 1 r S _ ] k Y A C ` Y ` Z I X ; k o ? P U R } 6 ~ @ + [ R D Y : C + a / 5 % 9 A _ j q V A / K 2 l R O C T ^ - U B  A 3 < 7  = ; Q ^ f P _    �    I  �  �  L  �  �  ~  ~  �  �  �  �  �  �  �  g  N  �  I  g  �  ,     Q  V  �  Y  �  �  !  �  `  �    �    q  P  �  s  �     U  �    �    j  �  *  v  �  `  �  �  I  Q  �    7    g  Q  �    ,    K  �  V  n     �  !  d  J  n  !  �  �  ==,1���
�ě�;�o;o:�o�ě��o��t��49X�49X�#�
��C��o��C�����D����C��,1�e`B�T�����ͽP�`���e`B���
�u��9X�+�#�
�49X���ͼ�`B���ͽo����h�C����q���D���o���H�9�,1��^5�m�h��㽏\)��w��O߽�\)�49X�,1�� Ž�hs�aG��49X�<j��o���
�<j�u�]/�Y��aG�����������o��+��\)����O߽��-��vɽ�7L��C���C����罸Q��/��"�BBa�B)��B2B.2BzBfzB�	BP`B��B�YB;B�yBw�B�QB&c B<�B��B!:rA��5A���B�XB�|B00A��fBVBrLBWBmB��B rB�&BsfB)<B+/KB/��B?�B��B��B=�B��B�|B$BB�B�$BʎB!pA��BO�B�B�B ��B��B)�GB-yuB�,B
�sB#$xB 4B��B�oB '
B|�BS�A��6B]�B �B%q3B
P[B�B�B�dB��BڔB	rB	�GB$�YB%�NB&�B
��BXAB�=B�hB;�BA�B)�B��B'�B��BC�B��B?�B� B��B@[B$VB�YB΅B&�sB�#B��B!?�A��A�~�B@�B�B��A�"�B@`BCBL�B��B/�B =�B4:B��B@�B*��B/�B>�B��B�MB�fB�wBƉB<�B�/B�	B��B!�sA��=BI|BǽB@�B A!B�cB*@TB-+�B�{B
�WB#2�B?RBE�B��B (8BC�B?�A��YB��B �
B%KeB
@�B<�B��B�B�~B�wB	�;B	�eB$O}B%}�B&0GB4.B33B�iB��C�:�A��xA)�hA�:A;��?��?�;@I�AGm"ABdA���@���A��A��@A�W�A�3�A�2�A��oAK*A���A�~bA�OPA�U�@��A�H�A�KoA��A�U$A�jAd|A�{B�A�R�@�!�Ao1-A\�A�?�A���@��r@eQAC�dA�!�Aw]�A���A-Ab�SA�?BA�B	�A�0�AgC@��A��@̓{AiyAЏ2B�H@9:�B
Z(Ayf�AݮS?�wA�>A��*A�A�&X@ؽ�@���A��A�C���>���C��~A�4bA���A¢nA,:�A%�A3�A�
G@�СA�>A��C�8dA�ݽA*�A�S�A;�L?��?�D�@I��AG�AB�dA��	@���A��A��pA�OuA���A�i�A�b>AM"�A��A�BA�uA�~�@��ZA�AƜYA��9A���A��Ab��AӆMBחA��C@�\oAnMuA[*�A�7�A�@�!-@ZAD�A���Av�JA�S�A,��Ad��A��BC�B	@oA�aAhq@���A�lc@Ѽ�A�bA�|	B�@5O;B
@�Az�@Aܧ�?�qA�t�A���A�~�A��@��@�'YA�bZA�sC���>��GC���A���A�`�A�A,A$�0A3�A��U@��A	8A�'                     
               (            G   )   	   "            )      -   	      
            
               
      
   $                  <         &      $   %   	      2   "               %         
      	   T                  J         #                                                         !            C   5      -            +      7               #   #            '   )            %   !               -                           9   )               #                  3   !                                                                                             ?   1      )            %                                    '               !                  '                           1                                                                                    N�,O_`�N��^N N�5?M��N��2N;m�N�E4N�GPNBp�OD�%O� Oe!vN�&P�Q�P@OPP�N�&�N5��OA�O���N�7O�÷N��N$��O�sO��lO��WOO*O��N��^N+��O�_�O �?N���O"��NVN�O�(�O�?�NB�Nr��N�oaOPFO��qN�}�N���O-�]O�GO�1%OxxNxp�N�.�P��O��O?�N#HQN0L�Oo�O�u{M��`N�{�NGrAN.IO7ݨO+�YN�\$O[P�O��N�3�O�ҬOx�N� �OOB�-NB�N>I N6zfO.`O,�N�"�O�9  �  �  j  �  �  J  l  �  %  x  ^  �    -  �  �  N  �  ?  d  �  i  �    t    �  ]  V  �  N    �  7  �  ?  B    �  +  5  �  "  n  �    �  �  }  �  �  �  �  A  o  �  �  �  z  �    �  �  �    �  	�    �  (      �  X  �  f  i    l  �  �  �  �=]/<�C�<49X<49X;�`B;D��;D��:�o�ě���o��o�D�����
���
�o�#�
�#�
�t��T���#�
�#�
�D����t��T�����D���T���e`B��o��t����
��o��C���t�������ͼ��
��1��1��h���ͼ��ͼ�`B��w���@��+�+�'+����P�t���P�,1�0 Ž���w�#�
�<j�]/�49X�8Q�8Q�@��@�����]/�L�ͽT���Y��Y���%�m�h�}󶽇+��o��o��o��7L�����Ƨ�Ƨ�������������������������������������������������������������������������������������������������������������������������$),))'& �������������HNS[_gnog_[QNFHHHHHH#/09/##)6BOPVXZ[OB<61)%$$#')).5BMLLJKGB5.)((''
#/48;93/#
���������������������#0U{�����{bI0�������
#/5745$�������������������������������������������������	!"	��������(/;C?;8/'#((((((((((tz������������zpoptt��������������*6BOXZUOB86,********SXamz}�����zmaTONNS6BOOWOONB96666666666GHNUYXVUHGECGGGGGGGG06=BGO[hrtytqh[OB630��������������������CHO[kpprqga_[OB;:>>Cz������������zwsqwz��������������������tz�������zvuvstttttt����������������������������������������*-67CFGC:6*������������?BNRV]glrsjg[NKHGD??��������������������������������� )59;:6)��� lnsz}~~znmjfllllllll$��!#/6881/#!!!!!!!!!X[h����~{xtrhc[YSRQX����������������������������������������kmz�������zummkkkkkk�������������������������������������������������������������������������#/16/'#������������������������� '*+��������'+6;BNTZZWXVSNB5)$$'|���������������zz||�����������������������

�������������QUWanz�����zvnaYURQQ������������������������������������������������������������15BBBA?>53)+11111111rz�����zwqrrrrrrrrrrXanz���������{~na_UX�������������������� #*0;<C<<80#t������������tonllnt��������������������./5<HPSSOH<7/.......Uanz��������zsmaURSU��

���������:BO[ghjhhc^[OKGB>6::V[fgt������{tg[TQQVVU[[gt��������tg[VROU
#%#
 #09840#           )0<>CIII<10)))))))))nt�������������ytpn������������������������������������������	 �������E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E��/�#��
����������
��#�/�<�D�H�F�@�<�/���������ĽȽнڽݽ������ݽнĽ��������U�R�O�P�U�a�c�b�a�]�U�U�U�U�U�U�U�U�U�U�M�I�A�:�4�(�$�(�,�4�A�M�Z�]�^�Z�W�P�M�M�3�-�'������'�3�4�9�3�3�3�3�3�3�3�3�3�2�0�3�9�@�G�L�X�Y�c�d�Y�Q�L�@�3�3�3�3��޺ֺҺֺ�������������������~�z�|������������������������f�]�Z�W�Z�`�f�s�z�����~�s�f�f�f�f�f�f�t�r�p�s�t�t�t�t�t�t�t�t�t�t�t�F�;�1�:�E�F�S�_�l�x�������~�x�q�l�_�S�F�������������������*�1�1�*�����������ſſ����������������������������������	���"�*�/�8�/�,�"���	���������m�\�R�@�*�/�d����������������������N�I�Q�Q�]�n����������������������|�Z�N����������������"�*�+�6�B�6�*�������������f�N�F�U�f������������׾��"� ���	��	��"�/�0�1�;�=�;�/�"�"�"�"�����"�)�/�7�/�"����������������������������������������������àÑÇ�n�a�Z�[�n�zÇÓâìÿ������ùìà�f�_�]�\�f�r�������|�r�f�f�f�f�f�f�f�f�N�A�4�'�%�(�5�A�Z�����������������s�g�N�a�[�a�a�n�z�}�z�u�n�a�a�a�a�a�a�a�a�a�a�����������������������������������������������������������������������������.�#�"�����/�;�T�\�a�g�a�W�T�N�H�;�.�"����	��"�.�;�T�`�m�s�����m�`�G�;�"��������������� ����)�)�'�"�������ƋƁ�h�d�f�h�sƁƎƚơƳ����ƿƿƳƧƚƋ�g�\�`�g�s�����������������s�g�g�g�g�g�g�M�D�@�4�3�4�8�@�B�M�T�Y�Z�Y�M�M�M�M�M�M���y�`�J�D�G�T�T�`�y�������ƿʿʿĿ������"��	����������	��"�'�1�;�B�;�.�"�[�S�O�J�O�Y�[�h�t��{�t�n�h�[�[�[�[�[�[�s�i�n�s�������������������������������s�@�9�>�@�G�M�Y�[�`�Y�Y�M�@�@�@�@�@�@�@�@�������~�d�]�_�]�f�~���������ºԺѺ������M�A�;�A�M�U�f�s��������������}�s�f�Z�M���������������������������������������˿������������ĿſʿɿĿ������������������<�8�1�<�H�U�a�d�e�a�U�H�<�<�<�<�<�<�<�<�нϽͽƽнݽ��������������ݽпG�;�"�	�������;�G�`�m�v�~���{�m�`�G�U�a�n�r�z�z�|�z�n�n�a�U�H�E�@�D�H�U�U�U����������������������������������������������� ���$�&�0�0�0�+�$��������������������������������������������T�G�;�.�*�#�$�.�5�;�T�`�m�z�����y�m�`�T�x�o�l�b�]�\�_�l�x���������������������x�6�5�,�5�6�B�K�O�Q�U�O�B�6�6�6�6�6�6�6�6��������'�4�@�M�U�M�D�@�4�,�'�����������ʼ����!�.�2�/�!������ʼ�������ùìææìù�������������������I�=�0�*�+�0�=�I�V�b�o�{�ǈǊǈ�{�o�V�I�ɺź������ɺֺ����ֺɺɺɺɺɺɺɺ��0�&�$�-�0�5�=�D�B�=�0�0�0�0�0�0�0�0�0�0�Ŀ��������������Ŀѿտۿݿ��ݿտѿĿ��h�V�L�Q�[�h�tāčĚĦĭĳĹİĦĚā�t�h�@�:�?�@�L�L�U�T�N�L�@�@�@�@�@�@�@�@�@�@�U�P�I�C�I�U�^�b�n�u�{ŇŇŃŀ�{�t�n�b�U����������������������������������������ĿĺĿ��������������ĿĿĿĿĿĿĿĿĿĿ�H�A�;�5�2�:�;�C�H�T�Y�_�_�a�m�t�g�a�T�H�@�@�4�-�.�4�8�@�M�Y�c�f�r�|�~�r�f�Y�M�@�����������������ûллܻ��޻ܻлû���Ŀ�������������
���#�%�#��
��������Ŀ������������������
��"�#�(�#���
����FFFFFFFF$F1F5F7F1F(F$FFFFFF�������������ùϹܹ����������ʹ�����EEEEE$E*E7ECEPEREYEZEPECE7E*EEEE��������������������������������������������������������������#� ��������#������#�-�/�<�A�H�K�N�Q�N�H�<�/�#�ݽѽ۽ݽ������������ݽݽݽݽݽݽݽݽ����������ĽɽνĽ������������������������� ������&�����������ŇņŇŋŊŔŜŠŭŹ����������ŹŭŠŔŇ�������x�x�������ûлܻ��߻ܻлû������!��!�"�.�.�:�G�J�M�G�@�:�.�!�!�!�!�!�!�S�H�L�S�[�`�l�y�������������|�y�l�`�S�S 4 ( 0 � : { 9 < " $ o 3 N 1 W P a ] ` Y A > d P K Z I X 0 _ V ? P U R _ 6 z @ ) U R D 9 : F + a  5 # : A _ a U V A / A B l R O C T 2  S B  A ( < #  = ; Q F f P _    �  �  �  �  �  L  �  P  �  �  �  �  F  �  
  ;  �  g  �  �  I  0  �  �  �  Q  V      V  �  !  �  `  �  �  �  �  q  �  P  s  �  �  U  c    �  r  j  W  �  v  �  �  `  �  I  Q  e    7    g  Q  �  k  	  �  K  �  V  @     O  �  d  J  n  �  �  �  =  C[  C[  C[  C[  C[  C[  C[  C[  C[  C[  C[  C[  C[  C[  C[  C[  C[  C[  C[  C[  C[  C[  C[  C[  C[  C[  C[  C[  C[  C[  C[  C[  C[  C[  C[  C[  C[  C[  C[  C[  C[  C[  C[  C[  C[  C[  C[  C[  C[  C[  C[  C[  C[  C[  C[  C[  C[  C[  C[  C[  C[  C[  C[  C[  C[  C[  C[  C[  C[  C[  C[  C[  C[  C[  C[  C[  C[  C[  C[  C[  C[  C[  C[  �  �  �  �  |  h  [  J  8  #    �  �  �    �  [  �  �  &  �  �  �  �  �  w  [  =    �  �  �  ~  O    �  �  W    n    +  K  b  i  c  S  =       �  �  �  h  E  ,  �  �  Z    �  �  �  �  {  v  q  j  d  ]  W  P  J  C  =  7  1  -  *  '  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  z  i  X  F  4  J  =  0  #       �   �   �   �   �   �   �   �   �   �   �   �   �   �  l  d  \  K  9  $    �  �  �  �  �  p  R  3    �  �  �  �  �  �  �  �  �  �  �  �  �  x  T  ,    �  �    Q  !  �  �  �  �  �      "  %  $         �  �  �  P    �  {  ;  �  m  k  h  `  Y  S  d  x  t  o  g  ]  P  7    �  �  �       ^    �  �  �  �  �  �  �  �  �  {  q  h  [  M  ;  *        "  C  a  x  �  �  y  s  x  ~  {  d  5  �  �  \  �  �  �                �  �  �  �  �  l  N  1    �  �  �  �  -    	  �  �  �  �  �  v  L    �  �  e  �  �     v  �  '  �  �  �  �  �  �  �  �  �  x  g  T  A  +        �  �  �  �  �  �  �  n  3  �  �  A  �  �  A  �  �  t    �  P  �   k  .  N  I  C  1        �  �  X  .  7    �  �  X  �  <  �  �  �  ~  q  d  _  s  �    r  c  S  B  -    �  �  �  r  ;  1  :  :  4     
  �  �  �  �  k  D  '  -    �  �  d  �     d  ]  U  N  F  >  2  '        �  �  �  �  �  �  �  �  o  �  �  �  �  �  �  �  �  �  �  ~  v  o  g  `  Y  Q  J  C  <  _  f  `  T  L  D  ;  D  �  �  �  �  �  ^  -  �  �  �  M    P  p  �  �  �  |  _  ?  $  X  M  !  =    �  z  �  �  �   �  �      
  �  �  �  �  �  �  l  6  �  �  ]    �  E  �  �  �  9  �  �  '  J  d  q  t  j  T  3    �  �  F  �  V  �  u              �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  Z  [  ]  V  M  D  ;  0  "    �  �  �  �  v  T  3     �   �  F  G  V  Q  G  :  *    �  �  �  �  u  K    �  �  c    �  �  �  �  �  �  �  �  �  z  d  W  ;    �  �  �  �  U    �  �  �  �  J  =  .  *  0  3  !  �  �  �  E  �  P  �  �  �   �          �  �  �  �  �  �  �  �  �  �  �  �  �  w  ;   �  �  �  �  y  n  a  Q  ?  ,    �  �  �  �  o  ;    �  �  n  7  0  )      �  �  �  �  �  �  �  l  R  -  �  �  �  i  :  �  �  w  h  Y  Y  ^  W  P  G  :  !    �  �  �  �  [     �  �  �  �      =  ?  8  .  %  8  /      �  �  �  �  g  w  B  :  2  )         �  �  �  �  �  s  F    �  �  �  R     �  �  �  �  �  �  �  c  D  :  2     �  �  �  F  �  �  @   �  �  �  �  �  �  �  �  o  ^  K  7  !  	  �  �  �  A  �  �  N      &  )        �  �  �  �  �  k  8  �  �    �    �  �  0    �  �  �  �  �  �  z  K    �  �  Y    �  �  s  p  �  �  �  �  �  �  �  �      6  J  =  1  #      �  �  �  "      �  �  �  �  �  �  j  J  (    �  �  �  R  #  �  �  )  S  N  C  8  4  8  ;  R  i  l  c  R  5    �  �  �  K  �  �  �  �  �  �  �  �  �  �  s  ]  F  &  	  �  �  �  {  L    �  �  �  �  �    �  �  �  �  �  �  �  |    �    ]  �   �  �  �  �  �  i  F  !  �  �  q  4  �  �  .  �  9  �  �  .   _  �  �  �  �  �  �  �  �  �  �  �  w  a  I  0    �  �  �  �    `  p  y  }  y  j  N  /    �  �  U    �  7  �  	  [  �  �  �  �  �  �    x  q  l  f  `  Z  T  J  <  -        �  �  �  �  �  �  �  �  g  ;    �  �  ^  '  �  �  T    �  �  �  �  �  �  �  �  l  F    �  �  �  \     �  �  n    �    �  �  �  �  �  �  �  �  n  U  ;       �  �  �  �  d  �  �  A  1  "      �  �  �  �  �  �  h  T  B  1       �   �   �  c  U  n  O  ,      �  �  �  e  &  �  �    �  ;  �    C  �  R  �  �  �  r  [  H  -  Q  f  X  B  "  �  �  �  C    5  �  �  �  �  {  m  ^  J  -    �  �  O  �  �  N  �  �  ?   �  �  �  �  �  �  �  �  {  m  `  T  G  5     
  �  �  �  �  �  z  y  x  v  r  m  g  \  M  >  $  �  �  �  �  k  D     �   �  |  �  �  �  �  �  y  c  E    �  �  �  �  W    �  �  �  �  b  �  �  �        �  �  �  �  �  x  g  D  �  U  �    �  �  �  �  �  �  �  �  �  �  �  �  �  r  ^  J  6  !    �  �  �  �  �  �  �  �  �  �  �  s  N    �  �  O  5    
  �  �  �  �  }  f  N  3    �  �  �  �  }  X  4    �  �  �  �  �    	  �  �  �  �  �  �  �  �  ~  p  b  W  P  I  C  >  9  4  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  {  _  D  �  �  �  �  	  	a  	�  	�  	�  	�  	�  	{  	@  �  �  �  �  A  i    �  �  �  �  �            �  �  �  �  �  �  T  #  �  �  �  �  �  �  �  �  �  �  �  �  �  |  b  @    �  k    �  !  (    �  �  �  �  �  _  8    �  �  �  �  R  	  �  {  5   �      �  �  �  �  �  o  >  
  �  �  k  3  �  �  �  M  !  
    �  �  �  �  �  �  u  X  <  "    �  �  �  R    �  a   �  �  �  �  �  �  �  u  2  �  �  )  �  �  C  t  ^  
  �  �  &  X  N  C  6  '      �  �  �  �  }  Y  6    �  �  �      r  �  �  �  �  �  �  �  �  p  L  !  �  �  |  0  �  �  L  �    <  X  e  f  d  _  Q  9    �  �  j    �  S  �  Z  �  �  i  f  b  _  \  X  U  P  K  E  @  :  5  0  +  &  !          �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  l  a  V  K  @  4  %      �  �  �  �  �  �  �  �  z  i  X  �  �  �  �  �  �  �  q  P  +    �  �  z  :  �  �    �  S  �  �  �  }  k  �  �  �  z  o  g  U  =  "    �  �  s  2  �  �  �  �  �  �  q  b  T  C  +    �  �  �  S  $  �  �  4  �  �  �  �  �  �  �  x  i  Y  H  8  )            !  L  {