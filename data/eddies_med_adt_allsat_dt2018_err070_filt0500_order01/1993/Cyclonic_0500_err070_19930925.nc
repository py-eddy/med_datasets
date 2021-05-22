CDF       
      obs    J   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type         ����         	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?`bM���   max       ?��1&�y     (  �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       M�sy   max       P�?�     (  �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       ���   max       =\)     (  �   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>Q��R   max       @F�p��
>     �  !$   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ���Q�    max       @v]�����     �  ,�   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @*         max       @Q@           �  8D   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @�        max       @�M`         (  8�   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       �<j   max       <�j     (  :    latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A��   max       B5)�     (  ;(   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A��   max       B4�      (  <P   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       <��+   max       C���     (  =x   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       =���   max       C���     (  >�   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          q     (  ?�   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          ?     (  @�   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          *     (  B   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       M�sy   max       P'3�     (  C@   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?��-V   max       ?�hr� Ĝ     (  Dh   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       �Ƨ�   max       =+     (  E�   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>Tz�G�   max       @F�p��
>     �  F�   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ���Q�    max       @v]�Q�     �  RH   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @         max       @Q@           �  ]�   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @�        max       @��@         (  ^l   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         >e   max         >e     (  _�   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?k�u%F   max       ?�[W>�6z        `�                                    7   7                        
            #                                       7   
            p                        %            :      ;             F      	   1         	      	            !   ZN�q�NeѱN%O�l�N�!!N�deN�q�N�PNt��N��qO8�'P �9Pw�=OR��P��O��N/�N�(OnoO��N��gO�*O��O��O�N.EN 8VN�aO�[1Og�]O�>�M�syNĲO��N&��N���OL5P�*OI�IN�{O�&IN��uP�?�NI��O��YO1xFO�A�N� "O�;tNqoAO�GOc�2ODN�G�P'3�N7�P�OSy�OJ��Oc�jPAIN��8O��O�MOH�O���N���N�,�O�N�ɬN���O�eN�!3PJ�=\)<���<D��<#�
;D��:�o:�o:�o:�o%@  ��o�D����o���
�ě��o�t��#�
�49X�49X�T���T���T����o��o��o��C���t���1��9X��9X��9X��j��j���ͼ��ͼ��ͼ��ͼ�/��/�����������+�,1�<j�<j�<j�@��L�ͽP�`�P�`�T���Y��]/�aG��ixսixսm�h�}󶽁%��������C���C���t����P���㽟�w���罩����,0:<ILPMIE<930*%,,,,��

������������./<FB><5//..........���)13#���������������������������#/;?=@;8/'&(########��������������������7BN[[a`[NB??77777777().6:=<?@61..,))((((OU]annrqniaUPLOOOOOO��������������������TF?<?DHTamz������zmTUez������������zpZUU����������������������������������&22 )Ngqx{tg[N5)#����������������������� �������������������		��������Z[cegpt��������tg[ZZ��������Z`ht�����������tkZUZ��*6CHJIEB6*��������������������������
"''"
��������chntt���~vtohbccccccaanz�~zna_aaaaaaaaaaQUadbaXUMPQQQQQQQQQQ$,6BLX\]UJB3ABHN[gt����tg[SKEEA�
#5<=AFFB</#
���EHRUWYUNIHFCEEEEEEEE)5BFNQNJGBA:52*)~��������������||zz~�����������������������������������������������������������������������#5BNgtz���g[NB5)$#���������������������������������������	

��������#0bm��yn]<��������������������#0<NbngihebUI<0
�������
����������%'#������������������������������������ ���������}����������z}}}}}}}}��������������������3<HUanxyxwtnaUHD<;73��������������������#/;<C></#)&$����������()`amsomiaa`WY````````�����!'*%��������xz|�����������zyvvvx)5=EHB5.)����������������������������������������[gpt������������tg[[_bdjn{������}zndb\_[ht���������xtlg[WV[Y[dgmt�������tg\[WWY���������������)57?BBB95)rty����������ytrrrrggt������������tgfcg��������������������TUY]aknpz}|znmfaUTTT������������������������������������������
/<EJMPTQTH/ ����������������ʼּ�����ּѼʼ����������f�^�f�n�s�����������s�f�f�f�f�f�f�f�fǈǅ�|ǈǔǡǧǡǙǔǈǈǈǈǈǈǈǈǈǈ��������������!�0�I�V�X�V�J�=�$��������g�c�^�g�s�������������s�g�g�g�g�g�g�g�g����������������������������������������O�M�I�O�[�h�n�t�z�t�h�[�O�O�O�O�O�O�O�O�M�G�H�M�M�Z�f�k�g�g�f�Z�M�M�M�M�M�M�M�M�@�>�@�F�M�Y�f�r�����r�f�Y�P�M�@�@�@�@�;�2�/�-�.�/�;�H�O�T�Y�Z�T�H�;�;�;�;�;�;�r�h�f�]�f�m�r�����������������������r�$�������ƧƠƌƈƉƒƧ�������$�'�)�$��ĳĜĚğĩĿ�����
�(�8�=�I�B�5�#�
��������������������������� ���������������*�������� ���*�\�h�oƀƁ�h�\�O�A�8�*�ѿĿ����������������Ŀ��������ݿѾ��������������������������������������������������������������������������������	����������	��"�/�9�;�G�I�@�;�/�"��	������������������������������������������s�r�s�w��������������������������Ľ������������нݽ���*�4�B�:����ݽĿG�;�+��	��������	��.�;�K�R�W�V�T�L�G�N�L�S�\�_�d�g�s�y�����������s�i�g�e�Z�N�������������������������	�������������x�q�l�_�_�_�_�`�l�x�y�������x�x�x�x�x�x�����������������������������������������������������������������������������������������m�c�i�y���������Ŀʿѿ߿ڿѿĿ��������������������������������������)����� ���5�B�N�[�`�\�U�N�C�7�5�)����������������������������������������������������������������������뾱�������������������ʾ۾����׾ʾ���¦©§¦ŔŏŋŐŔŠŭŹ��ŻŹŭŧŠŔŔŔŔŔŔ���׾־Ծ־���	�"�&�*�.�2�"���	����y�m�`�T�N�M�N�Y�`���������ĿۿݿͿ����y�����������	���������	���������z�z�m�i�b�m�z�����������������������x�l�b�Z�[�b�l�y�����������������������x����������}��������������ʾҾѾʾ������y�o�l�������о��4�W�s��s�A��н����y��������*�0�6�9�6�*����������{�c�Y�T�N�C�A�g�����������������������Ϲù������������������ùϹֹܹ޹��ܹϽ��������нݽ����4�?�K�H�<�7�'����Ľ�ŭŪŠŚŝŠŭŸŹ����������������ŹŭŭŹŭŤŢŨŭŹ��������������������Ź²©¦¡¦²·¿����¿²²²²²²²²�ɺ����������������ɺֺ���!�$�����ֺɹ������������������ùϹ۹ֹܹȹù��������!����
��"�.�;�G�T�Z�a�`�]�T�G�;�.�!���ݽ׽׽ݽݽ������������������!�:�`�r�|�������l�Q�:�!����������!������!�(�5�5�5�(��������������������ּ����!�,�+�%�����㼽�����Ŀ��������������Ŀѿۿ���������ݿѿ��O�J�B�6�0�6�B�H�]�tāčĒēčċ�~�t�h�O�	���������	��"�0�G�K�H�F�=�;�/�"��	�ܻлû��������ƻܻ���!�$��������ùóìëãàÛßàìùý����������üùù�û����������������ûлܻ����ܻػл�¦�²¿�����������������¿¦�#������%�/�7�<�H�J�O�O�M�H�G�<�/�#ÓÌÆÈÓàìíù��������������ùìàÓùñìåëìù������������������ùùùù����������������������������������������ĦĦģĤĦīįĳĿ����������������ĿĳĦ����������������������������������پʾȾ����������������ʾξ־׾����׾ʿ����������¿Ŀѿݿ�������߿ϿĿ��������������(�5�<�?�5�1�(������ED�D�D�D�D�EE*E7EPEiEuEyEmE\EPECE*EE @ ? D T F W E X � ' G K  a . > T P # ? . T Q O E � T n D 3 - p j ^ : = I   � N + x ^ 9 ; : ] ) N = j 2 B 8 - l ` 1 q ( , m , 9 1 Z " [ A H O \ F A  �  �  4  0  �  �  �  �  �  �  �    �  �  �  (  E  Q  �  D  �  r  p  l    g  P  *  �  �  /  -    R  L  �  �  Q  K  �  8  i  #  ]  <  �  �    O  n  �  �  �  �  �  ]  &  �  :  �  �  &  %    Y  3  �  �  :      �    �<�j<�j<#�
��C���o���
�ě���o�D�����
���ͽm�h�m�h��C����
�+�49X�T��������9X��9X�'�h�ě��P�`���㼛�㼣�
�0 Ž,1�D�����ͼ�����������P���
��P���D����P�hs�\)�L�ͽP�`��O߽]/��7L�T�����罛�㽇+�����G��m�h��l���\)���㽶E��J���-��t���l�����ě����㽟�w���Q콮{��"ѽ��<jB&.GB`;B�\B)5B.A���B�BhBq�B3B|*A��B �BIFB�B��B5)�B��A��B	�Bu�B��B/h�B�B>�B2�BηB�B�XB	ZB�3B��B<4B8�B�BU�B�B*�rBIDB0�B!z�B#�JB&L�BU	B&cAB=5BMlB�BNB7�B!�<B��BLB�B�:A��B-��B ��B��B �B��B
3_B(��B
`B	��B_�B�"B�XB
;,B:Bu�B�qB�$B|B&D�BuB�YB=bB6�A�fB4�B@5B��B7KBA�A�~�B�&B@DB�B��B4� B�bA��B
A9BD<B�JB/�B3B@]B�B|xB܃B��B	:5B҅B��B{�B9�B�B�\B�B*�$B�oB�B!<OB#:5B&?�BA/B&XzB.nB��BD7B@�B>0B!��B<EB��B>B�gA��B-��B ��BB�B!>�B��B
�B(�B
F�B	�&B,�B�B�B
�B=VB�1B�jB�\B?�@���AC��BqB	t�A���A��YA��A>��@܄�A��@�B�A�lA���B ;�Aw�ALZA�<�A�(A���AG�A+A_�WA�޿A��{@�P�A�wA�f\As�A�LA�9!A�JA��AO�'A��FA��vAY�9Ao�AZM�A���@�{�AKq4A'��A��FA�\>A NA/�+A��$A���A���@A

<��+AbN�A-8&A{�A��#Aw2A{�1A۠�A�FN@��NA�	I@�+[A��A��A͜�A��A���A�2gB��APm�Az�A��UC���@��XAC��B�B	�A���A��Aڀ�A>�@�9A�t�@�1B@7A��A�0B �Aw�AL��A��A��A�(�AF��A*�NA_ �A�4A�{�@���A�v�A���At��A�A��hA�r}A�j�AOr=A���A�AWQAn��AYpA��
@���AN�OA&�EA���A�F�=��}A3 	A��A�{�A��"@D�=���AaДA- �A"A�c�AjAz�AڻA���@��TA̋@��A�BA�}�Ä�A��A��ZA��B��AO�)Ax�A��
C���                                    8   7                        
         	   $                                       8               q               	         %            ;      <         !   F      
   1         	      	            !   [            %                        &   3      )   %                  )         #            #                           )               ?      )      )            *            *      -            +         %                              '            !                        !   #      )                     !                                                               !      #      )            #            *      )            )                                       'Nx�ANeѱN%O�ufN�!!N�deN�q�N�PNt��N��qO�O�4�P-%N�SP��O
�7N/�N�(OY��O��N��gO��O��N��{O}W0N.EN 8VN�aO�t�ODO���M�syNĲO��N&��N���OL5O���OI�IN�{O�*�N��uO��NI��O�IO1xFO�zhN� "O�;tNqoAO�K�Oc�2ODN�G�P'3�N7�O�OSy�O�cOW�P�N���O��O��PO�O���N���N�,�O�N�ɬN���O�eN�!3P��  :  �  �  �  �  w    �  7  :    �    .  �  t  "  �  �  �  Y  �  �  
    �  �  ?  �  �  �  r    @  �    �  �  �  1  �  �  �  A  �  *  �  �  y  �  �  I  {  �  �  C  �  1  :  
  	�  �  �  r  r  u  t  �    �  o  M  	8  =+<���<D��;��
;D��:�o:�o:�o:�o%@  ��o��o���ͼt��ě���1�t��#�
�D���49X�T������T����t���/��o��C���t����ͼě���j��9X��j��j���ͼ��ͼ��ͽ#�
��/��/������9X���o�+�0 Ž<j�<j�<j�]/�L�ͽP�`�P�`�T���Y��}�aG��u�m�h�y�#��%��%��hs��+��C���C���t����P���㽟�w���罩��Ƨ�)0<IINKI=<<50.))))))��

������������./<FB><5//..........��� )+*'��������������������������#/;?=@;8/'&(########��������������������7BN[[a`[NB??77777777().6:=<?@61..,))((((OU]annrqniaUPLOOOOOO��������������������DHTamz����~zmaTNGDADmz������������zmfdhm������������������������������������@BGNX[gjnmig[NEB?@@@����������������������� �������������������		��������Z[cegpt��������tg[ZZ��������hrt����������thc\]`h��*6CHJIEB6*���������������������������

��������chntt���~vtohbccccccaanz�~zna_aaaaaaaaaaQUadbaXUMPQQQQQQQQQQ!).36BFQVWTPE6LN[gt����}sg[UNLGGDL�

#/<<AFEB</#
��EHRUWYUNIHFCEEEEEEEE)5BFNQNJGBA:52*)~��������������||zz~�����������������������������������������������������������������������#5BNgtz���g[NB5)$#���������������������������������������	

��������!#(0<COW\ab`UI<0," !��������������������#0<IbhehfcUI<0�������
���������$&"������������������������������������ ���������}����������z}}}}}}}}��������������������3<HUanxyxwtnaUHD<;73��������������������#/;<C></#)&$����������()`amsomiaa`WY````````�����	$$�������xz|�����������zyvvvx)+25;BB<53)!
����������������������������������������`gst����������tg````_bdjn{������}zndb\_Z^clt���������tg`\ZZZ[gt}�������tg^[XXZZ���������������)57?BBB95)rty����������ytrrrrggt������������tgfcg��������������������TUY]aknpz}|znmfaUTTT������������������������������������������
/<CFKQROKH/# ��������������ʼּܼ���ּ̼ʼ������������f�^�f�n�s�����������s�f�f�f�f�f�f�f�fǈǅ�|ǈǔǡǧǡǙǔǈǈǈǈǈǈǈǈǈǈ�������������,�0�=�I�R�T�P�E�=�������g�c�^�g�s�������������s�g�g�g�g�g�g�g�g����������������������������������������O�M�I�O�[�h�n�t�z�t�h�[�O�O�O�O�O�O�O�O�M�G�H�M�M�Z�f�k�g�g�f�Z�M�M�M�M�M�M�M�M�@�>�@�F�M�Y�f�r�����r�f�Y�P�M�@�@�@�@�;�2�/�-�.�/�;�H�O�T�Y�Z�T�H�;�;�;�;�;�;�r�m�f�e�f�q�������������������������r��ƴƝƘƝƬ���������� �������������ĺĳıĶļ���������������������������������������������������������������*�������� ���*�\�h�oƀƁ�h�\�O�A�8�*�������������������ĿѿԿڿؿѿ˿Ŀ��������������������������������������������������������������������������������������	����������	��"�/�7�;�E�G�>�;�/�"��	������������������������������������������s�r�s�w��������������������������������������Ľݽ����������ݽнĽ��G�;�+��	��������	��.�;�K�R�W�V�T�L�G�Z�U�X�Z�`�e�g�s�~���������s�g�^�Z�Z�Z�Z����������������������������������������x�q�l�_�_�_�_�`�l�x�y�������x�x�x�x�x�x�����������������������������������������������������������������������������������������y�m�j�s�y�����������Ŀ̿ԿѿĿ��������������������������������������)���������5�B�N�Z�_�[�T�N�A�5�)����������������������������������������������������������������������뾱�������������������ʾ۾����׾ʾ���¦©§¦ŔŏŋŐŔŠŭŹ��ŻŹŭŧŠŔŔŔŔŔŔ���׾־Ծ־���	�"�&�*�.�2�"���	����m�`�W�V�Z�`�m�����������������������y�m�����������	���������	���������z�z�m�i�b�m�z�����������������������x�l�c�[�\�c�l�x�����������������������x����������}��������������ʾҾѾʾ������������������������нݽ��������Ľ�����������*�0�6�9�6�*����������d�[�U�R�G�N�g�������������������������Ϲù������������������ùϹֹܹ޹��ܹϽ������н߽�����4�=�J�F�;�6�%����н�ŭŪŠŚŝŠŭŸŹ����������������ŹŭŭŹŭŤŢŨŭŹ��������������������Ź²©¦¡¦²·¿����¿²²²²²²²²�����������ĺɺֺ����������ۺֺɺ��������������������ùϹ۹ֹܹȹù��������!����
��"�.�;�G�T�Z�a�`�]�T�G�;�.�!���ݽ׽׽ݽݽ������������������!�:�`�r�|�������l�Q�:�!����������!������!�(�5�5�5�(��������������������˼ּ����!�'�&�!������ּ��Ŀ��������������Ŀѿۿ���������ݿѿ��O�C�C�K�O�[�b�tāąčďčĈā�{�t�h�[�O��	�� �����	��"�/�;�F�H�J�H�D�;�/�"��ܻлû��������ɻ����!���������ùõììäàÛààìùû������ûùùùù�û����������������ûлܻ����ܻػл�¿¦¦²¿��������������¿�#�!���#�&�/�8�<�H�I�N�N�L�H�C�<�/�#�#ÓÌÆÈÓàìíù��������������ùìàÓùñìåëìù������������������ùùùù����������������������������������������ĦĦģĤĦīįĳĿ����������������ĿĳĦ����������������������������������پʾȾ����������������ʾξ־׾����׾ʿ����������¿Ŀѿݿ�������߿ϿĿ��������������(�5�<�?�5�1�(������ED�D�D�D�D�EE*ECEPEiEuEvElEdEPECE*EE C ? D R F W E X � ' D F  E .  T P ! ? . > Q Q @ � T n ? 0 ' p j ^ : = I  � N ) x ? 9 3 : Z ) N = X 2 B 8 - l V 1 f ( & n , , 4 Z " [ A H O \ F ?  �  �  4  �  �  �  �  �  �  �  c  ,  6    �  ,  E  Q  �  D  �  g  p  �    g  P  *  8  �    -    R  L  �  �  U  K  �    i  w  ]  
  �  s    O  n  �  �  �  �  �  ]  `  �    �  �  )  %  �  <  3  �  �  :      �    �  >e  >e  >e  >e  >e  >e  >e  >e  >e  >e  >e  >e  >e  >e  >e  >e  >e  >e  >e  >e  >e  >e  >e  >e  >e  >e  >e  >e  >e  >e  >e  >e  >e  >e  >e  >e  >e  >e  >e  >e  >e  >e  >e  >e  >e  >e  >e  >e  >e  >e  >e  >e  >e  >e  >e  >e  >e  >e  >e  >e  >e  >e  >e  >e  >e  >e  >e  >e  >e  >e  >e  >e  >e  >e    (  2  9  :  9  5  .  '      �  �  �  �  �  w  [  @  %  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  t  S  +  �  �  �  B  �    4  �  �    |  w  s  n  g  `  W  N  D  7  *    �  �  �  �  �  w  o  g  _  S  G  :  -         �  �  �  �  �  �  �  �  �        �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  A  �  �  �  �  �  �  �  �  �  �  �  x  g  I  !   �   �   �   y   M   !  7  ,       	  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  :  /  $        �  �  �  �  �  �  �  p  S  6     �   �   �  �  �    �  �  �  �  �    Y  -  �  �  L  �  Y  �  H  �  =  �  �  �  �  �  �  �  �  �  f  %  �  s  �  t  �  ^  �      ,  ~  �  �  �  �  �      �  �  �  B  �  n    �  <  �  `  )  "          (  $       �  �  �  �  w  C    �  {  Q  �  f  N  =  5  ,  "    *  4  A  O  P  E  1    �  �  �  �  [  d  T  V  Q  K  H  P  a  p  t  p  `  F    �  �  L  �  .  "                     �   �   �   �   �   �   �   �   �   y  �  �  �  �  �    {  y  x  w  v  u  t  r  l  g  b  \  W  R  �  �  �  �  �  �  z  l  ]  L  7    �  �  �  ~  L    �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  v  M  "  �  �  '  Y  Q  I  :  +      �  �  �  �  �  y  \  8    �  �  [    m  �  �  �  �  �  �  �  �  �  �  �  �  W    �  s     �    �  �  �  �  o  [  G  3      �  �  �  �  u  M  (  �  �  c  �      	  	  
           �  �  �  �  �  �  v  F   �   �  j  �  �  �              �  �  �  T    �  �  S  �   �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  ~  {  w  t  p  l  i  e  b  ^  ?  E  K  R  X  ^  d  j  p  v    �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  ~  Z  +  �  �  �  G  �     �  �  �  �  �  �  �  �  �  d  D    �  �  �  d  )  �  �  .  �  �  �  �  �  �  �  �  �  v  c  C    �  �  r  4  �  �  �  r  u  w  y  {  }    �  �  �  �  �  �  �  �  �  �  �  �  �    �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  @  8  1  )  "        �  �  �  �  �  ~  b  E  (   �   �   �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  |  j  W  D         �  �  �  �  �  �  �  �  ~  h  O  1    �  �  �  y  �  �  �  �  �  �  ~  m  ^  g  g  `  U  H  7  "    �  �    b  �  �  �  �  �  �  �  �  �  �  q  A    �  f  �  >  ~  >  �  z  n  b  U  F  6  "    �  �  �  �  �  �  v  �  �  �  �  1  )           �  �  �  �  �  �  �  �  �  �  �         �  �  �    m  N  #  �  �  �  �  �  z  o  [  C  )    �  $  �  �  o  Y  B  +      �  �  �  �  �  w  ^  I  3     �   �  -  C  4  ;  L  Q  A  7  {  �  �  }  J  �  s  �  $  �  A    A  <  6  1  +  &  "              �  �  �  �  �  �  �  �  �  �  �  �  |  h  R  ;       �  �  �  R    �  �  B    *        �  �  �  �  �  l  F    �  �  �  j  3  �  �  `  �  �  �  �  �  |  T  ,    �  �  �  �  =  �    +  �  �    �  �  �  �  �  �  �  �  t  b  Q  @  .      �  �  �  �  �  y  _  B  !  �  �  �  �  �  �  w  Y  4    �  �  >  �  �    �  {  q  h  ]  Q  E  6  %      �  �  �  �  �  �  V  (   �  �  �  �  �  �  �  �  s  R  &  �  �  s  ,  �  �  F    �  V  I  "  �  �  �  {  S  "  �  �  �  `  1  �  �  m  	  �  �    {  f  R  >  (    �  �  �  �  �  Y  /    �  �  �  e  >    �  �  �  �  N  +    �  �  v  >    �  �  b  %  �  �  +  �  �  }  q  o  p  a  J  +  �  �  T  �  �  �    �    F  �  �  C  1      �  �  �  �  �    V  ,    �  �  �  k  V  A  ,  �  �  �  �  �  �  �  �  j  7  �  �  :  �    �  �     V  %  1    �  �  �  �  �  i  K  -  
  �  �  �  �  c  S  G  >  :  �  	  &  :  '  	  �  �  �  n  @    �  �  1  �  t    �  �  �      �  �  �  �  �  �  �  �  s  D    �  Y  �  �  �   �  	�  	�  	�  	~  	5  �  �  1  �  �  5  �  �  n  +  �  Q  �  �  ~  �  �  �  �  �  �  �  Y  *  �  �  �  V    �  �  N  �  :  S  �  �  �  �  �  �  �  �  �  ~  i  Q  8      �  �  �  O    6  _  o  q  j  ^  M  9  !    �  �  s  1  �  _  �  �    ;  \  q  m  e  T  <    �  �  �  p  -  �  �  \    �  r  �    u  U  3  *    �  �  �  ~  Z  2    �  �  q  ;    �  �    t  p  k  b  V  H  5  #    �  �  �  �  �  x  c  O  C  D  E  �  w  ]  C  (    �  �  �  �  �  c  F  0  '        
        �  �  �  �  �  �  �    `  A  "    �  �  �  �  q  T  �  �  �  �  d  L  6  #    �  �  �  �  �  �    j  S  6    o  \  H  3      �  �  �  �  �  �  v  e  X  N  E  ?  9  4  M  5    �  �  �  �  �  P    �  �  I  �  �  :  �  �  n  O  	8  	#  	  �  �  �  �  �  m  >     �  .  �  *  �    �  �  O  �  	  �  �  t     �  l  2     !  "    
�  
p  	�  	  �  L  f