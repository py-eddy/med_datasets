CDF       
      obs    K   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type         ����         	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?`bM���   max       ?�"��`A�     ,  �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       M���   max       Pr�%     ,  �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       ���   max       <�`B     ,      effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?8Q��   max       @Fq��R     �  !0   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ���G�|    max       @vhQ��     �  ,�   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @$         max       @Q�           �  8�   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @ɖ        max       @�C@         ,  98   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       �C�   max       �t�     ,  :d   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A���   max       B1f0     ,  ;�   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A���   max       B0��     ,  <�   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       >lXz   max       C� U     ,  =�   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       >I�   max       C�&$     ,  ?   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          j     ,  @@   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          5     ,  Al   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          +     ,  B�   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       M���   max       P,}�     ,  C�   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?��J�M   max       ?�I�^5?}     ,  D�   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       ���   max       <��
     ,  F   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?@        max       @Fp��
=q     �  GH   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       �ٙ����    max       @vg�
=p�     �  S    speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @         max       @Q�           �  ^�   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @ɖ        max       @�          ,  _P   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         E�   max         E�     ,  `|   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�kP��{�   max       ?�I�^5?}     p  a�   :         j   +                  Q   -      	            	      '   %   ?            ?   #      )            H         <                        (   	            -         5   	   	                                                 	               #O�6'NΚ�Ot�qP/$�P�O��O5�O���N8��Ocf�Pr�%P��OUN$(_N	�O�ۊN���N���N>\O��P��O�\�N"��N�4pOgiP�XOWΣO]��O�2M���N)<WM�nMO�;�N�gNl|Px�N²�N�9N���N��YN��N���N��!O�09N�~Nb�Np��N�ʂP,}�O�xBO+�~Oc>O�sN:dNF��N��|NC��N��N�[AN��O��N��N�N���NFe�O��aN�|NCZ/Nq��N��O�,pO=ÇN5��N��O��<�`B<o��o��o�o�o���
�ě��#�
�#�
�T���e`B�e`B�u��C����
���
��1��1��1�ě����ͼ��ͼ��ͼ�����h��h��h���+�+�t��t��t���P��P��P��w�'8Q�8Q�8Q�8Q�<j�@��aG��aG��e`B�ixսixսixսm�h�m�h�u�u�y�#��%��%��o��o��������7L��7L��O߽�O߽�O߽�hs���㽛�㽟�w���w���
�������������������������������������������#/<HUahmla]UH<.# )BOWhorqdd[OC6-##/Halva]SPH</#
�����������������}}9;HTZagllaTH;942999928BLO[gt�����tgNI912������������������������������������ 
�������������BORm���������h[OB>AB|��������������|y{||lnz����znjllllllllll��������������������
������������
W[agptutrmgf[STSWWWW:BGN[fgmqg[NB8::::::
 	







#*4C\hu���uh\OC6*$!#����������������������������������TUanoonea\WUTTTTTTTTlnstxz��������ztnjklTVabhmpomha]TQONMMMT���������	
��������O[ht|}xzytlh[OJDCEIO���������������������������
#((
���������������������������������������������;;;HHJH;:8;;;;;;;;;;sw|����������������s��������������������stz����������tssssssCHUa���������xaUH?=C��������������������MOP[[hpt~tpoh[ZUSOM�����������zz}���������}zzzzzzz�������������������������������������������

����������������������������	#)3676/)				HIKU\bhbUPIGHHHHHHHH��������������������jn{������|{yponmjjjAOgt����������g[ND>A[gt������ztlga[VRRT[suz���������������us")5CLQPNB5)#$'.0<IPUXYURI=<0$���������������������������������������
#*/130/#
��xz��������zyxxxxxxxx������������������������������������������������������������N[gt~������|tg[WNIIN�����������������������
!%#!
�����#%',..'#����������������������%%"�����������������������������������������������������������!,BN[adb[N5)%)5BMNQNB5)"���������������������� ����������!%(5BN[^\YUN5)��
���������������
��<�H�N�T�Q�F�4�/��������������ûлܻ�������ܻлû������)��������)�6�?�B�F�Q�X�O�H�B�6�)�F�-�� �:�G�S�l�����ûۻ޻лͻû����l�F�J�9�3�-�9�A�M�i�s������������������s�J�(�������(�4�:�A�M�M�P�M�M�A�4�(�(�h�f�b�d�h�o�uƁƎƓƙƚƐƎƁ�u�h�h�h�h���Ӿʾʾ־�������	�
� ����	����A�9�9�@�A�M�S�U�R�M�A�A�A�A�A�A�A�A�A�A�-�(��������!�-�:�F�J�R�Q�M�C�:�-�ʾ��������׿	�;�T�`�m���~�m�`�.�"�	��ʿM�I�;�"����.�y���������������}�m�`�M�Z�R�N�D�C�N�U�Z�g�s�y�����������s�g�Z�Z�H�G�@�?�H�U�V�V�V�U�H�H�H�H�H�H�H�H�H�H��ݾ۾������ ���������������e�r�~�����������x�r�e�Y�F�J�F�<�C�L�f�e���������������������ÿĿпĿ������������f�`�Z�S�P�Z�\�f�s�z��~�w�s�f�f�f�f�f�f�H�?�H�T�V�a�m�q�m�m�a�T�H�H�H�H�H�H�H�H�ʾ����������������ʾ���
��������ʿ����u�l�n�����������������˿����Ŀ����Ŀ����������Ŀѿ�����	��������ݿ��m�l�d�j�m�z�|�������z�q�m�m�m�m�m�m�m�m�;�6�/�"����"�/�:�;�C�H�J�O�T�X�T�H�;���������	��"�/�2�;�E�H�S�H�;�/�"����������ܻڻ�����4�X�W�T�@�4�'���L�F�H�T�Y�f�r�����������������r�f�Y�L�{�f�Y�T�I�K�M�Y�f�r������������������{������������������������������������������������
�����
���������������������H�F�<�2�<�<�H�S�U�V�U�Q�H�H�H�H�H�H�H�H�A�A�A�M�N�Z�g�\�Z�N�A�A�A�A�A�A�A�A�A�AĚā�t�h�.���)�B�h�tāčĚĦĳĸĳĠĚ���������������ûƻû»������������������O�L�C�:�6�6�6�6�C�G�O�V�V�R�O�O�O�O�O�O���������������ùܹ�����������Ϲ����M�M�I�M�X�Z�f�q�s�u�������s�f�f�Z�M�M�Ϲɹù����������ùϹ׹ܹ��������ܹԹϼ���ݼ޼�������������������������������������������������������������������Ƶ��������������������������(�5�A�N�W�N�M�A�5�(�������ѿ̿ʿϿѿֿݿ��������������ݿѿ���������������*�=�C�L�O�Y�O�6�*���Z�S�M�F�M�O�Z�b�f�i�s�z�|�u�s�f�Z�Z�Z�Z�:�7�-�+�-�3�:�E�C�F�N�F�:�:�:�:�:�:�:�:�'�&��'�4�@�M�T�M�H�@�4�'�'�'�'�'�'�'�'�S�N�O�S�]�_�l�v�x�����������x�s�l�_�S�S��²�v�u�z²����������������������������������0�B�[�g�j�[�Q�B�)����V�I�=�4�0�%�0�4�=�I�J�V�b�o�o�p�o�k�b�V�R�O�I�O�S�\�h�tāčĎĕĚĘēčā�t�h�R����������������������������������������D�D�D�D�D�EEEEEED�D�D�D�D�D�D�D�D������������������������ܻû����������������ûȻлڻܻ������������������������������������������������������������������������������������������������������*�8�6�*�&��������\�R�O�I�O�\�h�uƁƆƂƁ�u�h�\�\�\�\�\�\�����
���#�&�0�<�U�\�Z�N�I�?�0�#������������������������������������EEEE%E*E0E7ECEPEZE\EiE_E\EPECE;E7E*EE�E�E�E�EuElEuE�E�E�E�E�E�E�E�E�E�E�E�E�ŔŒŎŒŔŕŠŭűŰŭŠŔŔŔŔŔŔŔŔ�������{�������Ľнݽ����������ݽн����G�>�;�G�S�`�b�l�t�l�`�Y�S�O�G�G�G�G�G�G�ֺԺɺƺźɺֺߺ����ֺֺֺֺֺֺֺּ�߼ݼ޼�����������������������������������������������������}�s�l���������������������������������}�(�!�������(�5�C�N�X�Z�c�Z�N�A�5�(�b�V�\�b�n�{�}�~�{�n�b�b�b�b�b�b�b�b�b�b�������������������	�����������������ìàÓÇ�z�n�\�S�O�Q�U�n�zàëðõ��ùì % ~ C H D 1 ( Q K 6 T *   b J a V > T 9 f  + N j &  O < J G t V X e 7 L 5 ^ - f f 7 D * S ; R D a - : J v < f S X e e ! G a v 0 Y � @ ~ r b ? 3 e ^    �  J  �  )  Z  .  M  v  r  �  *  �  /  b    O  �    f    s  ~  B  �  �  \  �  �  �    O  +  �  �  �  �  �    -  �  N  �  ,    �  9  w  �  P  �  r  �  Q  ^  g  k  l  �  I  �  
    &  �  b  Y  F  P  �  �  G  �  =  �  ����t���`B����49X���
���ͼ�1�e`B�#�
����m�h��/��j��1�H�9�+�������q���u��9X��`B��`B�,1��j��%�L�ͽ�hs�\)�8Q��w��/�0 Ž0 Žě��L�ͽ]/�Y��m�h����T���u�� ŽaG��ixսy�#�}�����-��\)��S���7L��C���C������C���\)���-��\)�Ƨ𽝲-�\�� Ž��-��^5��hs���㽲-��{������"ѽ�9X��l��C�B�B�B�B1�B�Bi�A��B��B�+B!y�B�IB��B_BS�B ��B#b�B	�Bv�BZ�B1f0B+#zB��BA�BT�A���BBR�B Y�BBp�B �zA���B�lBD�B
��B��B�B��B-�xB �wB ��B`�BeB+B�0B'HB)� B(�B
+�B	�B
�wB�B&G�Bh�B#5B$��B \xBN�BB)sB	.�B�B��B�OB�B�mB��BN�B+�BkB*�B�B}{B)Be�B��B�"BķB��BD9B��A�}�BɛBRZB!UBԽB�'BywB}AB �DB#��B	A�B�.BD�B0��B*��B��BB�B��A�t�BC�B)�B �AB<�B��B �.A���B˕B>�B
��B�B��Bd�B.2"B �QB ��BE�B{�BB6B��B'qlB)�FB(� B	�B	� B
��B;�B&zBA�B"ӳB$��B EB�KB4,B��B	6 B3qB�|B�(B��B?;B%BEeBcHB�lB7�B̤BL
BD�B�&A�>@�A�]@��VA@X�A8�zB�AX��A;ǅ@s�'AZw�Aj�LA��.A���AV��@˄Au_�AA��A���ASIAu�GA}��A�#�A�@5A�\�@ƭ�@�Z�@�u�A���A��"A�] A��tA��@��_B �Z>lXzAAgX>���A��A҃�BVA��A~*�A���A@��@~)7@�W�@��3A���A��pB��A�{�A�$gC�M�A.�9@�>�A夾A�XuA�uBI�A��oBp+C��cC� UA�YIA%�{A�F@<��A��A��A���A���A�[�Aѹ{Aɫ|A��@��CA׀!@���A?rA9 B�AY�A;S@t�sAX��AkֶA�"A�muAV�N@�aAs��AA�^A�g�AT��Av�pA}$0A�L�A��bA��d@��@ᙸ@�M A���A��mA��A���A�Ξ@�4�B ȵ>I�A@�B>�ɢAPA��B�8A���A~aA�}�A@��@y(�@�B>@�;�A���A�NB��A�h�A��C�FJA/V@���A�A�A���B>�A��BmNC��=C�&$A�A&��A�'@?,�A��A	GA���A��]A�|zAю/Aǐ&   ;         j   +                  R   -      	            
      '   %   @            @   #      )            I         <                        )   	            -         6   
   	   	                  !            	               
         	      $            /   %                  5   )            &            %   3               %         !            )         %                                       +                                                                  %            !            %                     #                           %   %               #                              %                                       +                                                                  %            !O���Nke�Ot�qO�k	OrH?O��N��#O���N8��OO���O5N�2�N$(_N	�O;��N���N���N>\Oݏ8O��ROy^%N"��N�4pOG�O��SO)f�O"7#O���M���N)<WM�nMO�T�N�gNl|P	��N�g�N�9N���N��YNͩ�M�\\N�V&OPBN�~Nb�Np��N�ʂP,}�O�O+�~OPݱO�sN:dNF��N��|NC��N��N�[AN��O:�N��N�Ne�aNFe�O��aN�|NCZ/Nq��N��O�,pO-�N5��N��O��  	c  #  	  	�  a  R  �  `       A  �  m  �  �  �  �  X  e  �  �  	  �  �  �  f  �  U    �  �    
�  5  3  p  W  K  �  �  �  �  t  G  $  H  $  r    �    b  \  �  �  +  �  �  �  7  �  �  
0  f  �  �  i  V  C  �  �    �  �  �<��
;D����o��`B����o�o��`B�#�
��t��<j��w��C��u��C���󶼣�
��1��1��9X��h��P���ͼ��ͼ�/���C��o���+�+�t��e`B�t���P�#�
����w�'8Q�@��L�ͽ<j�T���@��aG��aG��e`B�ixսq���ixսy�#�m�h�u�u�y�#��%��%��o��o���P�����7L��hs��O߽�O߽�O߽�hs���㽛�㽟�w���
���
�������������������������������������������#/<HUahmla]UH<.# $)6BO[diki][OB;8.!"$#,/<HSYYQLH</#���������������}}5;<HTVabhdaTH<;6555555;BN[dty����~tgNK:5����������������������������������������������������ehltv����������tshee~�������������|~~~~~lnz����znjllllllllll�������������������������
��������W[agptutrmgf[STSWWWW:BGN[fgmqg[NB8::::::
 	







$+6COhu���uh\OC6*%"$����������������������������������������TUanoonea\WUTTTTTTTTlnstxz��������ztnjklNTabglonmfa[TRPONNNN����������������KOS[huvtstoh]ZOKFEGK�������������������������
"!
�����������������������������������������������;;;HHJH;:8;;;;;;;;;;����������������������������������������stz����������tssssssDHUa���������naUH@>D��������������������MOP[[hpt~tpoh[ZUSOM�����������zz}���������}zzzzzzz�������������������������������������������

�����������������������������	#)3676/)				HIKU\bhbUPIGHHHHHHHH��������������������jn{������|{yponmjjjAOgt����������g[ND>AZgt������|xtg]XUTSUZsuz���������������us )25BKPNB5)#$'.0<IPUXYURI=<0$���������������������������������������
#*/130/#
��xz��������zyxxxxxxxx������������������������������������������������������������LNP[^gp~|vtmgd[RNLL�����������������������
!%#!
�����#),+%#����������������������%%"�����������������������������������������������������������!,BN[adb[N5)')5BKNPNLB54)���������������������� ����������!%(5BN[^\YUN5)��
�����������
��#�<�H�P�M�A�<�5�/�#����������������ûлܻ��ܻлû����������)��������)�6�?�B�F�Q�X�O�H�B�6�)�F�=�9�C�D�J�S�_�x�������ûû������l�S�F�f�Z�M�H�A�?�;�:�A�M�Z�f�s���������s�f�(�������(�4�:�A�M�M�P�M�M�A�4�(�(�u�i�h�d�g�h�s�uƁƎƐƖƗƎƍƁ�u�u�u�u�����ؾ;پ�������	�������	���A�9�9�@�A�M�S�U�R�M�A�A�A�A�A�A�A�A�A�A�-�%�!������!�-�:�?�F�J�L�G�F�:�1�-��׾ѾȾľ˾׾���	��;�H�I�=�.�"�	����m�g�`�W�T�H�K�T�`�g�m�y�����������}�y�m�Z�V�N�J�M�N�Z�g�s�|���������s�g�Z�Z�Z�Z�H�G�@�?�H�U�V�V�V�U�H�H�H�H�H�H�H�H�H�H��ݾ۾������ ���������������Y�V�Z�^�i�r�~�������������������~�r�e�Y���������������������ÿĿпĿ������������f�`�Z�S�P�Z�\�f�s�z��~�w�s�f�f�f�f�f�f�H�?�H�T�V�a�m�q�m�m�a�T�H�H�H�H�H�H�H�H�ʾ����������������ʾ�����������ʿ����{�r�t�����������Ŀѿ���ݿѿĿ����Ŀÿ������ĿͿѿݿ����	�	�����ݿѿ��m�l�d�j�m�z�|�������z�q�m�m�m�m�m�m�m�m�;�6�/�"����"�/�:�;�C�H�J�O�T�X�T�H�;���������	��"�/�0�;�C�H�P�H�;�/�"��	��������߻�������4�V�V�S�@�4�'���Y�Q�M�H�K�Y�f�l�r���������������r�f�Y������r�f�Y�V�K�M�N�Y�f�s�������������������������������������������������������������
�����
���������������������H�F�<�2�<�<�H�S�U�V�U�Q�H�H�H�H�H�H�H�H�A�A�A�M�N�Z�g�\�Z�N�A�A�A�A�A�A�A�A�A�A�f�[�O�<�'�(�6�B�O�[�h�tāęěđčā�t�f���������������ûƻû»������������������O�L�C�:�6�6�6�6�C�G�O�V�V�R�O�O�O�O�O�O�������������ùܹ��������������Ϲ����Z�O�M�L�M�Z�Z�f�s�������s�f�d�Z�Z�Z�Z�Ϲɹù����������ùϹ׹ܹ��������ܹԹϼ���ݼ޼����������������������������������������������������������������������
�	������������������������'�(�5�A�C�A�A�5�(���������ѿο˿пѿ׿ݿ������������ݿѿѿѿ���������������� �*�9�C�F�H�C�@�*��Z�S�M�F�M�O�Z�b�f�i�s�z�|�u�s�f�Z�Z�Z�Z�:�7�-�+�-�3�:�E�C�F�N�F�:�:�:�:�:�:�:�:�'�&��'�4�@�M�T�M�H�@�4�'�'�'�'�'�'�'�'�S�N�O�S�]�_�l�v�x�����������x�s�l�_�S�S��²�v�u�z²����������������������������������-�5�B�S�Z�N�B�5�)����V�I�=�4�0�%�0�4�=�I�J�V�b�o�o�p�o�k�b�V�Z�O�L�O�T�[�]�h�tāčēĘėĒčā�t�h�Z����������������������������������������D�D�D�D�D�EEEEEED�D�D�D�D�D�D�D�D������������������������ܻû����������������ûȻлڻܻ������������������������������������������������������������������������������������������������������*�8�6�*�&��������\�R�O�I�O�\�h�uƁƆƂƁ�u�h�\�\�\�\�\�\�#����
��
��#�0�<�B�I�K�I�F�<�7�0�#�����������������������������������EEEE%E*E0E7ECEPEZE\EiE_E\EPECE;E7E*EE�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�ŔŒŎŒŔŕŠŭűŰŭŠŔŔŔŔŔŔŔŔ�������{�������Ľнݽ����������ݽн����G�>�;�G�S�`�b�l�t�l�`�Y�S�O�G�G�G�G�G�G�ֺԺɺƺźɺֺߺ����ֺֺֺֺֺֺֺּ�߼ݼ޼�����������������������������������������������������}�s�l���������������������������������}�(�#������"�(�/�5�B�M�N�V�Z�N�A�5�(�b�V�\�b�n�{�}�~�{�n�b�b�b�b�b�b�b�b�b�b�������������������	�����������������ìàÓÇ�z�n�\�S�O�Q�U�n�zàëðõ��ùì # � C = . 1 ( K K ; G $  b J Y V > T 9 ;  + N h & & W ( J G t X X e 4 M 5 ^ - Y t 6 8 * S ; R D W - 7 J v < f S X e e  G a V 0 Y � @ ~ r b ? 3 e ^�N  ,  �  �  �  �  .    "  r  D  �  7  �  b    �  �    f     9  �  B  �  o  0  b  �  P    O  +  9  �  �  |  �    -  �    H    �  �  9  w  �  P  ,  r  �  Q  ^  g  k  l  �  I  �  J    &  l  b  Y  F  P  �  �  G  �  =  �  �  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  �  	;  	`  	a  	P  	5  	  �  �  �  L     �     �  2  �  [  J  R  �                      0  P  �  �  �  �  �  �  w  X  	  �  �  �  �  �  �  �  s  ^  T  @    �  �  4  �  L  v   �  �  �  	<  	~  	�  	�  	�  	p  	~  	�  	�  	  	m  	O  	  �      w  �  �  �  �    &  B  V  _  a  V  ?    �  �  8  �  v  �  	  �  R  H  >  .            !    �  �  �  i  D    �  }  �  n  �  �  �  �  �  p  W  5    �  �  ~  <  �  �  c    �  �  R  ]  \  X  T  S  T  Q  H  8  "    �  �  q  5      6  4             �  �  �  �  �  �  �  �  �  �  �  �  �  �    �  �  �  �  �  �  �  �  �  �  �  �  V    �  �  3  �  �  �  �  <  �  (  �  �  !  ;  @  .  	  �  �  {    �  �    B  �  �  �  "  N  {  �  �  �  �  �  �  �  �  �  m  3  �  �  c    0  C  U  d  j  l  j  c  V  E  /    �  �  �  t  G    �  �  �  �  �  �  �  s  G      �  �  �  �  �  o  S  7    �  �  �  �  �  �  �  �  �  �    o  ^  L  :  '       �   �   �   �  H  F  `  g  i  q    �  }  j  T  >    �  �  �  a  (  �  h  �  �  �    l  W  A  *    �  �  �  �  �  r  B    �    u  X  U  R  Q  O  R  V  Z  ]  ^  [  Y  V  S  Q  N  K  I  P  W  e  V  G  7  (      �  �  �  �  �  {  z  z  z  z  {  |  }  �  �  �  �  �  �  l  M  ,  
  �  �  �  �  W    �  h  �  
  M  O    �  �  v  f  Q  5    �  �  �  r  7  �  �  Q  �   �  {  �  �  	  	  �  �  �  �  \  )  �  �  �  U  �  Y  �  *  0  �  �  �  �  �  �  �  �  �  �  ~  w  p  j  f  b  _  [  X  T  �  ~  u  m  d  \  S  K  D  =  6  .  '      	   �   �   �   �  �  �  �  �  �  �  �  �  �  �  }  f  P  @  2    �  �  A  �  `  b  Y  G    �  �  �  �  �  �  �  �  �  d  �  I  �  �  u  k  �  �  �  �  m  O  '  �  �  �  K    �  h    �  :  �  m    A  S  U  P  @  -      =  =  .          #      �  �            �  �  �  �  N    �    #  �  ;  �  �    �  �    7  a  �  �  �    2  �  4  �  o  	  	�  
H  
�  �  !  �  �  �  �  �  ^  0    �  �  s  A    �  �  `  &  �  �  m                      "  )  /  0  %        �  �  	�  
$  
W  
�  
�  
�  
�  
�  
�  
�  
�  
�  
f  
#  	�  	H  �  �  �  w  5  )        �  �  �  �  �  �  �  �  o  Z  F  2    �  �  3  .  (  #        �  �  �  �  �  �  �  j  @    �  �  e  ^  o  [  8    �  �  s  C    �  �  B  �  �  ,  �  $  ~  C  4  L  V  V  U  T  S  R  P  L  H  B  9  *    �  �  �  �  �  K  F  <  .      �  �  �  �  �  �  �  �  �  q  Y  <  !    �  �  �  �  z  i  W  F  5  #    �  �  �  t  .  �  �  2   �  �  �  m  U  F  &  �  �  �  �  W  +  �  �  �  `  +  �  �  �  �  �  �  �  �  �  �  m  D    �  �  �  U    �  |  "  �  7  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  #  �    a  p  n  e  [  N  >  +    �  �  �  q  8  �  �  s  .  �  �  �    (  G  B  .  	  �  �  P    �  {  ,  �      �  �  �  $    
  �  �  �  �  �  �    j  T  >  &    �  �  �  �  �  H  H  G  G  F  F  F  E  E  D  C  @  >  ;  9  6  4  1  /  ,  $      �  �  �  �  �  �  �  z  d  M  5        �   �   �   �  r  m  g  b  V  I  =  )    �  �  �  �  �  �  �  �  �  o  [    �  �  �  �  �  �  L    �  ~  4  �  �  c    �  #  [  �  �  �  �  �  �  i  _  `  B    �  �  X    �  \    �  S      �  �  �  �  �  �  j  K  *    �  �  �  R    �  w     �  Z  b  ^  L  /    �  �  �  ^     
�  
a  	�  	T  �  �    K  ]  \  T  L  D  =  6  .  &        �  �  �  �  �  �  n  O  /  �  �  m  E    �        0  e  u  ^  G  2      �  �  �  �  �  �  �  �  �  �  �  �  �  t  f  V  F  3  !    �  �  %  K  +    �  �  �  �  �  �  �  �  �  �  �  �  �  �  m  J  &     �  �  �  �  �  q  W  >  "     �  �  �    a  B  *    �  �  �  �  �  �  �  �  �  �  |  b  H  /    �  �  �  Z  "  �  �  �  �  �  �  �  �  �  �  �  _  A    �  �  �    G  �  u  �  7  ,  !            
  �  �  �  �  �    &  <  (      5  �  �  �  �  �  �  �  �  m  0  �  �     �  >  �  =  �  �  �  �  �  \  ?  "    �  �  �  �  n  S  7    �  �  �  T    
0  
  
  	�  	�  	�  	�  	�  	�  	T  	  �  �  A  �  �  S    �  k  P  0    +  f  D    �  �  �  F  �  �  ;  �  u    �  ;  �  �  �  �  �  �  �  �  �  �  �  �  �  u  m  d  m  |  �  �  �  �  �  �  �  �  �  �  q  ^  E  '    �  �  x  >  �  K  M   �  i  Y  H  8  (      �  �  �  �  �  �  �  �  �  �  �  �  �  V  F  6  &      �  �  �  �  �  �  �  e  H  +     �  �  v  C    �  �  �  �  e  ;    �  �  s  >  	  �  �  [  �  �  �  �  �  �  �  �  �  �  p  ^  R  J  ;  &    �  �  �  �  �  Y  �  z  a  B  "     �  �  �  �  e  B  '  	  �  �  �  8  �  W        �  �  �  �  �  X    �  �  >  �  |    �    \    �  �  �  �  �  �  �  o  W  =  !    �  �  �  �  _  :    �  �  s  M  $  �  �  �  c  +  �  �  f  %  �  �  E  �  L  �  w  �  �  }  E      �  �  D  �  �  k    �  c  �  V  �  �  5